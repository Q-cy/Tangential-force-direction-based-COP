const SENSOR_ROWS: usize = 12;
const SENSOR_COLS: usize = 7;
const SENSOR_COUNT: usize = SENSOR_ROWS * SENSOR_COLS;

const CONTACT_ENTER_TOTAL_THRESHOLD: f32 = 520.0;
const CONTACT_ENTER_PEAK_THRESHOLD: f32 = 50.0;
const CONTACT_EXIT_TOTAL_THRESHOLD: f32 = 260.0;
const CONTACT_EXIT_PEAK_THRESHOLD: f32 = 28.0;
const CONTACT_ENTER_FRAMES_REQUIRED: usize = 2;
const CONTACT_EXIT_FRAMES_REQUIRED: usize = 8;

const BASELINE_IDLE_ALPHA: f32 = 0.035;
const BASELINE_BOOTSTRAP_ALPHA: f32 = 1.0;
const BASELINE_NOISE_FLOOR: f32 = 5.0;

const ACTIVE_CELL_MIN_VALUE: f32 = 18.0;
const ACTIVE_CELL_PEAK_RATIO: f32 = 0.14;
const MIN_ACTIVE_CELLS: usize = 3;

const ANCHOR_LERP_ALPHA: f32 = 0.018;
const VECTOR_SMOOTHING_ALPHA: f32 = 0.16;

const REPORT_MAGNITUDE_ENTER: f32 = 0.12;
const REPORT_MAGNITUDE_EXIT: f32 = 0.045;
const REPORT_CONFIDENCE_ENTER: f32 = 0.14;
const REPORT_CONFIDENCE_EXIT: f32 = 0.06;
const REPORT_HOLD_FRAMES: usize = 10;

const ASYMMETRY_WEIGHT: f32 = 1.1;
const DRIFT_WEIGHT: f32 = 0.65;
const MOTION_WEIGHT: f32 = 0.25;

#[derive(Debug, Clone, Copy)]
pub struct PztSpatialAnalysis {
    pub angle_deg: f32,
    pub magnitude: f32,
    pub planar_x: f32,
    pub planar_y: f32,
    pub confidence: f32,
    pub contact_active: bool,
    pub reportable: bool,
}

pub struct PztProcessor {
    baseline_frame: Option<Vec<f32>>,
    contact_active: bool,
    contact_enter_counter: usize,
    contact_exit_counter: usize,
    anchor_cop_x: Option<f32>,
    anchor_cop_y: Option<f32>,
    last_cop_x: Option<f32>,
    last_cop_y: Option<f32>,
    smoothed_x: f32,
    smoothed_y: f32,
    report_active: bool,
    report_hold_counter: usize,
    held_report: Option<PztSpatialAnalysis>,
}

#[derive(Clone, Copy)]
struct ContactStats {
    total: f32,
    peak: f32,
    active_total: f32,
    active_cells: usize,
    min_row: usize,
    max_row: usize,
    min_col: usize,
    max_col: usize,
    cop_x: f32,
    cop_y: f32,
    asymmetry_x: f32,
    asymmetry_y: f32,
}

impl PztProcessor {
    pub fn new() -> Self {
        Self {
            baseline_frame: None,
            contact_active: false,
            contact_enter_counter: 0,
            contact_exit_counter: 0,
            anchor_cop_x: None,
            anchor_cop_y: None,
            last_cop_x: None,
            last_cop_y: None,
            smoothed_x: 0.0,
            smoothed_y: 0.0,
            report_active: false,
            report_hold_counter: 0,
            held_report: None,
        }
    }

    fn reset_tracking_state(&mut self) {
        self.contact_active = false;
        self.contact_enter_counter = 0;
        self.contact_exit_counter = 0;
        self.anchor_cop_x = None;
        self.anchor_cop_y = None;
        self.last_cop_x = None;
        self.last_cop_y = None;
        self.smoothed_x = 0.0;
        self.smoothed_y = 0.0;
    }

    fn reset_report_state(&mut self) {
        self.report_active = false;
        self.report_hold_counter = 0;
        self.held_report = None;
    }

    fn update_idle_baseline(&mut self, raw_frame: &[f32], alpha: f32) {
        match self.baseline_frame.as_mut() {
            Some(baseline) => {
                for (base, current) in baseline.iter_mut().zip(raw_frame.iter().copied()) {
                    *base += (current - *base) * alpha;
                }
            }
            None => {
                self.baseline_frame = Some(raw_frame.to_vec());
            }
        }
    }

    fn subtract_baseline(&mut self, raw_frame: &[f32]) -> Vec<f32> {
        if self.baseline_frame.is_none() {
            self.update_idle_baseline(raw_frame, BASELINE_BOOTSTRAP_ALPHA);
        }

        let baseline = self
            .baseline_frame
            .as_ref()
            .expect("baseline should exist after bootstrap");

        raw_frame
            .iter()
            .zip(baseline.iter())
            .map(|(raw, base)| (raw - base - BASELINE_NOISE_FLOOR).max(0.0))
            .collect()
    }

    fn pressure_metrics(frame: &[f32]) -> (f32, f32) {
        let total = frame.iter().sum::<f32>();
        let peak = frame.iter().copied().fold(0.0, f32::max);
        (total, peak)
    }

    fn is_contact_enter_frame(frame: &[f32]) -> bool {
        let (total, peak) = Self::pressure_metrics(frame);
        total >= CONTACT_ENTER_TOTAL_THRESHOLD && peak >= CONTACT_ENTER_PEAK_THRESHOLD
    }

    fn is_contact_exit_frame(frame: &[f32]) -> bool {
        let (total, peak) = Self::pressure_metrics(frame);
        total <= CONTACT_EXIT_TOTAL_THRESHOLD || peak <= CONTACT_EXIT_PEAK_THRESHOLD
    }

    fn inactive_analysis() -> PztSpatialAnalysis {
        PztSpatialAnalysis {
            angle_deg: 0.0,
            magnitude: 0.0,
            planar_x: 0.0,
            planar_y: 0.0,
            confidence: 0.0,
            contact_active: false,
            reportable: false,
        }
    }

    fn weak_contact_analysis() -> PztSpatialAnalysis {
        PztSpatialAnalysis {
            contact_active: true,
            ..Self::inactive_analysis()
        }
    }

    fn compute_contact_stats(frame: &[f32]) -> Option<ContactStats> {
        let total = frame.iter().sum::<f32>();
        if total <= 0.0 {
            return None;
        }

        let peak = frame.iter().copied().fold(0.0, f32::max);
        if peak <= 0.0 {
            return None;
        }

        let active_threshold = (peak * ACTIVE_CELL_PEAK_RATIO).max(ACTIVE_CELL_MIN_VALUE);

        let mut active_total = 0.0;
        let mut active_cells = 0usize;
        let mut weighted_col_sum = 0.0;
        let mut weighted_row_sum = 0.0;
        let mut min_row = SENSOR_ROWS;
        let mut max_row = 0usize;
        let mut min_col = SENSOR_COLS;
        let mut max_col = 0usize;

        for row in 0..SENSOR_ROWS {
            for col in 0..SENSOR_COLS {
                let index = row * SENSOR_COLS + col;
                let value = frame[index];
                if value < active_threshold {
                    continue;
                }

                active_cells += 1;
                active_total += value;
                weighted_col_sum += value * col as f32;
                weighted_row_sum += value * row as f32;
                min_row = min_row.min(row);
                max_row = max_row.max(row);
                min_col = min_col.min(col);
                max_col = max_col.max(col);
            }
        }

        if active_cells < MIN_ACTIVE_CELLS || active_total <= 0.0 {
            return None;
        }

        let cop_x = weighted_col_sum / active_total;
        let cop_y = weighted_row_sum / active_total;
        let bbox_center_x = (min_col + max_col) as f32 * 0.5;
        let bbox_center_y = (min_row + max_row) as f32 * 0.5;
        let half_width = ((max_col - min_col).max(1) as f32) * 0.5;
        let half_height = ((max_row - min_row).max(1) as f32) * 0.5;

        let mut asymmetry_x = 0.0;
        let mut asymmetry_y = 0.0;

        for row in min_row..=max_row {
            for col in min_col..=max_col {
                let index = row * SENSOR_COLS + col;
                let value = frame[index];
                if value < active_threshold {
                    continue;
                }

                asymmetry_x += value * ((col as f32 - bbox_center_x) / half_width);
                asymmetry_y += value * ((row as f32 - bbox_center_y) / half_height);
            }
        }

        Some(ContactStats {
            total,
            peak,
            active_total,
            active_cells,
            min_row,
            max_row,
            min_col,
            max_col,
            cop_x,
            cop_y,
            asymmetry_x: asymmetry_x / active_total,
            asymmetry_y: asymmetry_y / active_total,
        })
    }

    fn compute_vector_angle(x: f32, y: f32) -> (f32, f32) {
        let magnitude = (x * x + y * y).sqrt();
        if magnitude <= f32::EPSILON {
            return (0.0, 0.0);
        }

        let mut angle = y.atan2(x).to_degrees();
        if angle < 0.0 {
            angle += 360.0;
        }

        (angle, magnitude)
    }

    fn update_contact_state(&mut self, raw_frame: &[f32], frame: &[f32]) -> bool {
        if self.contact_active {
            if Self::is_contact_exit_frame(frame) {
                self.contact_exit_counter += 1;
                if self.contact_exit_counter >= CONTACT_EXIT_FRAMES_REQUIRED {
                    self.update_idle_baseline(raw_frame, BASELINE_IDLE_ALPHA);
                    self.reset_tracking_state();
                    self.reset_report_state();
                    return false;
                }
            } else {
                self.contact_exit_counter = 0;
            }

            return true;
        }

        if Self::is_contact_enter_frame(frame) {
            self.contact_enter_counter += 1;
            if self.contact_enter_counter >= CONTACT_ENTER_FRAMES_REQUIRED {
                self.contact_active = true;
                self.contact_enter_counter = 0;
                self.contact_exit_counter = 0;
                return true;
            }

            return false;
        }

        self.contact_enter_counter = 0;
        self.update_idle_baseline(raw_frame, BASELINE_IDLE_ALPHA);
        false
    }

    fn store_report(&mut self, mut analysis: PztSpatialAnalysis) -> PztSpatialAnalysis {
        analysis.reportable = true;
        self.report_active = true;
        self.report_hold_counter = 0;
        self.held_report = Some(analysis);
        analysis
    }

    fn hold_or_drop_report(&mut self) -> PztSpatialAnalysis {
        if self.report_active && self.report_hold_counter < REPORT_HOLD_FRAMES {
            self.report_hold_counter += 1;
            if let Some(mut held) = self.held_report {
                held.reportable = true;
                return held;
            }
        }

        self.reset_report_state();
        Self::weak_contact_analysis()
    }

    fn stabilize_report(&mut self, analysis: PztSpatialAnalysis) -> PztSpatialAnalysis {
        if !analysis.contact_active {
            self.reset_report_state();
            return analysis;
        }

        let can_enter = analysis.magnitude >= REPORT_MAGNITUDE_ENTER
            && analysis.confidence >= REPORT_CONFIDENCE_ENTER;
        let can_stay = analysis.magnitude >= REPORT_MAGNITUDE_EXIT
            && analysis.confidence >= REPORT_CONFIDENCE_EXIT;

        if self.report_active {
            if can_stay {
                return self.store_report(analysis);
            }

            return self.hold_or_drop_report();
        }

        if can_enter {
            return self.store_report(analysis);
        }

        analysis
    }

    pub fn get_pzt_analysis(
        &mut self,
        adc_data: &[f32],
    ) -> Result<PztSpatialAnalysis, &'static str> {
        if adc_data.len() != SENSOR_COUNT {
            return Err("ADC data length must be 84");
        }

        let baseline_subtracted = self.subtract_baseline(adc_data);
        if !self.update_contact_state(adc_data, &baseline_subtracted) {
            return Ok(Self::inactive_analysis());
        }

        let Some(stats) = Self::compute_contact_stats(&baseline_subtracted) else {
            return Ok(self.stabilize_report(Self::weak_contact_analysis()));
        };

        let Some(anchor_x) = self.anchor_cop_x else {
            self.anchor_cop_x = Some(stats.cop_x);
            self.anchor_cop_y = Some(stats.cop_y);
            self.last_cop_x = Some(stats.cop_x);
            self.last_cop_y = Some(stats.cop_y);

            return Ok(self.stabilize_report(Self::weak_contact_analysis()));
        };
        let anchor_y = self.anchor_cop_y.unwrap_or(stats.cop_y);
        let last_x = self.last_cop_x.unwrap_or(stats.cop_x);
        let last_y = self.last_cop_y.unwrap_or(stats.cop_y);

        let drift_x = stats.cop_x - anchor_x;
        let drift_y = stats.cop_y - anchor_y;
        let motion_x = stats.cop_x - last_x;
        let motion_y = stats.cop_y - last_y;

        let combined_x = stats.asymmetry_x * ASYMMETRY_WEIGHT
            + drift_x * DRIFT_WEIGHT
            + motion_x * MOTION_WEIGHT;
        let combined_y = stats.asymmetry_y * ASYMMETRY_WEIGHT
            + drift_y * DRIFT_WEIGHT
            + motion_y * MOTION_WEIGHT;

        self.smoothed_x += (combined_x - self.smoothed_x) * VECTOR_SMOOTHING_ALPHA;
        self.smoothed_y += (combined_y - self.smoothed_y) * VECTOR_SMOOTHING_ALPHA;

        self.anchor_cop_x = Some(anchor_x + drift_x * ANCHOR_LERP_ALPHA);
        self.anchor_cop_y = Some(anchor_y + drift_y * ANCHOR_LERP_ALPHA);
        self.last_cop_x = Some(stats.cop_x);
        self.last_cop_y = Some(stats.cop_y);

        let planar_x = self.smoothed_x;
        let planar_y = -self.smoothed_y;
        let (angle_deg, magnitude) = Self::compute_vector_angle(planar_x, planar_y);

        let active_span_rows = (stats.max_row - stats.min_row + 1) as f32 / SENSOR_ROWS as f32;
        let active_span_cols = (stats.max_col - stats.min_col + 1) as f32 / SENSOR_COLS as f32;
        let activity = (stats.active_cells as f32 / SENSOR_COUNT as f32).clamp(0.0, 1.0);
        let span = ((active_span_rows + active_span_cols) * 0.5).clamp(0.0, 1.0);
        let pressure_ratio = (stats.active_total / stats.total.max(1.0)).clamp(0.0, 1.0);
        let peak_ratio =
            (stats.peak / (stats.total / stats.active_cells as f32 + 1.0)).clamp(0.0, 1.0);
        let confidence =
            ((activity * 0.35) + (span * 0.2) + (pressure_ratio * 0.3) + (peak_ratio * 0.15))
                .clamp(0.0, 1.0);

        Ok(self.stabilize_report(PztSpatialAnalysis {
            angle_deg,
            magnitude,
            planar_x,
            planar_y,
            confidence,
            contact_active: true,
            reportable: false,
        }))
    }

    pub fn get_pzt_angle(&mut self, adc_data: &[f32]) -> Result<f32, &'static str> {
        Ok(self.get_pzt_analysis(adc_data)?.angle_deg)
    }

    pub fn should_report(analysis: &PztSpatialAnalysis) -> bool {
        analysis.reportable
    }

    pub fn reset_baseline(&mut self) {
        self.baseline_frame = None;
        self.reset_tracking_state();
        self.reset_report_state();
    }
}

#[cfg(test)]
mod tests {
    use super::{PztProcessor, SENSOR_COLS, SENSOR_ROWS};

    fn index(row: usize, col: usize) -> usize {
        row * SENSOR_COLS + col
    }

    fn make_frame(active: &[(usize, usize, f32)]) -> [f32; SENSOR_ROWS * SENSOR_COLS] {
        let mut frame = [0.0; SENSOR_ROWS * SENSOR_COLS];
        for (row, col, value) in active {
            frame[index(*row, *col)] = *value;
        }
        frame
    }

    #[test]
    fn idle_frame_does_not_report_contact() {
        let mut processor = PztProcessor::new();
        let frame = [0.0; SENSOR_ROWS * SENSOR_COLS];
        let analysis = processor.get_pzt_analysis(&frame).unwrap();
        assert!(!analysis.contact_active);
        assert!(!analysis.reportable);
        assert_eq!(analysis.magnitude, 0.0);
    }

    #[test]
    fn right_heavy_contact_reports_rightward_angle_after_confirmation() {
        let mut processor = PztProcessor::new();
        let baseline = [0.0; SENSOR_ROWS * SENSOR_COLS];
        let contact = make_frame(&[
            (5, 2, 120.0),
            (5, 3, 180.0),
            (5, 4, 280.0),
            (6, 2, 110.0),
            (6, 3, 170.0),
            (6, 4, 260.0),
            (7, 2, 100.0),
            (7, 3, 150.0),
            (7, 4, 240.0),
        ]);

        let _ = processor.get_pzt_analysis(&baseline).unwrap();

        let mut analysis = processor.get_pzt_analysis(&contact).unwrap();
        for _ in 0..8 {
            analysis = processor.get_pzt_analysis(&contact).unwrap();
        }

        assert!(analysis.contact_active);
        assert!(analysis.reportable);
        assert!(analysis.magnitude > 0.0);
        assert!(analysis.angle_deg <= 45.0 || analysis.angle_deg >= 315.0);
    }

    #[test]
    fn report_stays_active_through_short_weak_gap() {
        let mut processor = PztProcessor::new();
        let baseline = [0.0; SENSOR_ROWS * SENSOR_COLS];
        let contact = make_frame(&[
            (5, 2, 120.0),
            (5, 3, 180.0),
            (5, 4, 280.0),
            (6, 2, 110.0),
            (6, 3, 170.0),
            (6, 4, 260.0),
            (7, 2, 100.0),
            (7, 3, 150.0),
            (7, 4, 240.0),
        ]);
        let weak = make_frame(&[(5, 3, 55.0), (5, 4, 60.0), (6, 3, 50.0), (6, 4, 58.0)]);

        let _ = processor.get_pzt_analysis(&baseline).unwrap();
        for _ in 0..10 {
            let _ = processor.get_pzt_analysis(&contact).unwrap();
        }

        let analysis = processor.get_pzt_analysis(&weak).unwrap();
        assert!(analysis.reportable);
    }
}

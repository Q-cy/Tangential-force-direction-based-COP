"""完整 API 示例：显式保留采集循环，具体功能由 package 提供。"""

try:
    from tangential.full import (
        FullAcquisitionSession,
        FullApplicationConfig,
        FullApplicationRunner,
        g_main_stop_flag,
    )
except ModuleNotFoundError:
    from src.tangential.full import (
        FullAcquisitionSession,
        FullApplicationConfig,
        FullApplicationRunner,
        g_main_stop_flag,
    )


def data_loop(
    plot,
    stop_event=None,
    config=None,
    session_factory=FullAcquisitionSession,
    **session_kwargs,
):
    """完整采集循环；每轮只编排会话公开方法。"""
    session = session_factory(
        plot,
        config=config or FullApplicationConfig(),
        stop_event=stop_event or g_main_stop_flag,
        **session_kwargs,
    )
    try:
        session.start()
        while not session.should_stop():
            session.check_errors()
            session.process_new_pressure_frames()
            session.drain_force_matches()
            session.log_timing_stats()
            session.update_plot()
            session.wait_for_next_iteration()
    finally:
        session.close()


def main():
    FullApplicationRunner(data_loop).run()


if __name__ == "__main__":
    main()

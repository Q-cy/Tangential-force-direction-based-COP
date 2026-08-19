"""最小 API 示例：压力采集、CoP、角度、梯度和标定。"""

from tangential_package import FixedTerminalRenderer, TangentialSensorAPI


def main():
    renderer = FixedTerminalRenderer()
    try:
        with TangentialSensorAPI() as sensor:
            while True:
                sample = sensor.read(timeout_s=0.1)
                if sample is not None:
                    renderer.render(sample)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

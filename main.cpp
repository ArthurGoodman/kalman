#include <chrono>
#include <iostream>
#include <Eigen/Geometry>
#include <QtCore/QDebug>
#include <QtCore/QTimer>
#include <QtGui/QFontDatabase>
#include <QtGui/QKeyEvent>
#include <QtGui/QPainter>
#include <QtGui/QPainterPath>
#include <QtWidgets/QApplication>
#include <QtWidgets/QWidget>
#include <opencv2/tracking/kalman_filters.hpp>

namespace {

static constexpr double c_epsilon = 1e-4;

static constexpr double c_max_speed = 200 * 3.6;
static constexpr double c_max_acceleration = 40 * 3.6;
static constexpr double c_max_wheel_angle = M_PI / 5;

static constexpr double c_vehicle_wheel_base = 3;
static constexpr double c_vehicle_width = 2;

using Vector2d = Eigen::Matrix<double, 2, 1, false>;

template <class T>
cv::Mat structToCvMat(const T &data)
{
    cv::Mat mat(sizeof(data) / sizeof(double), 1, CV_64F);
    for (int i = 0; i < mat.rows; i++)
    {
        mat.at<double>(i, 0) = *(reinterpret_cast<const double *>(&data) + i);
    }
    return mat;
}

template <class T>
T cvMatToStruct(const cv::Mat &mat)
{
    T data;
    for (int i = 0; i < mat.rows; i++)
    {
        *(reinterpret_cast<double *>(&data) + i) = mat.at<double>(i, 0);
    }
    return data;
}

struct UkfState
{
    Vector2d position;
    double speed;
    double heading;
};

struct UkfControl
{
    double acceleration;
    double yaw_rate;
};

struct UkfMeasurement
{
    Vector2d position;
    double speed;
    double heading;
};

class CUkfSystemModel final : public cv::tracking::UkfSystemModel
{
public: // methods
    void measurementFunction(
        const cv::Mat &x_k,
        const cv::Mat &n_k,
        cv::Mat &z_k) override
    {
        UkfState state = cvMatToStruct<UkfState>(x_k);

        UkfMeasurement z = cvMatToStruct<UkfMeasurement>(x_k);

        z.position = state.position;
        z.speed = state.speed;
        z.heading = state.heading;

        cv::Mat mat = structToCvMat(z) + n_k;
        mat.copyTo(z_k);
    }

    void stateConversionFunction(
        const cv::Mat &x_k,
        const cv::Mat &u_k,
        const cv::Mat &v_k,
        cv::Mat &x_kplus1) override
    {
        UkfState state = cvMatToStruct<UkfState>(x_k);
        UkfControl control = cvMatToStruct<UkfControl>(u_k);

        UkfState new_state;

        const double w = control.yaw_rate;
        if (std::abs(w) <= c_epsilon)
        {
            const Eigen::Rotation2Dd R{state.heading};
            new_state.position =
                state.position +
                R *
                    Vector2d{state.speed + control.acceleration * m_dt / 2,
                             0.0} *
                    m_dt;
        }
        else
        {
            const double a = control.acceleration;
            const double v = state.speed;
            const double dt = m_dt;
            const double alpha = state.heading;

            const double delta_x =
                1.0 / w / w *
                ((v * w + a * w * dt) * std::sin(alpha + w * dt) +
                 a * std::cos(alpha + w * dt) - v * w * std::sin(alpha) -
                 a * std::cos(alpha));
            const double delta_y =
                1.0 / w / w *
                ((-v * w - a * w * dt) * std::cos(alpha + w * dt) +
                 a * std::sin(alpha + w * dt) + v * w * std::cos(alpha) -
                 a * std::sin(alpha));

            new_state.position = state.position + Vector2d{delta_x, delta_y};
        }

        new_state.speed = state.speed + control.acceleration * m_dt;

        new_state.heading = state.heading + control.yaw_rate * m_dt;

        cv::Mat mat = structToCvMat(new_state) + v_k;
        mat.copyTo(x_kplus1);
    }

    void setDeltaTime(double dt)
    {
        m_dt = dt;
    }

private: // fields
    double m_dt;
};

struct VehicleState
{
    Vector2d position;
    double speed;
    double acceleration;
    double heading;
    double wheel_angle;
};

class CVehicleController final
{
public: // methods
    explicit CVehicleController()
        : m_state{}
    {
        m_state.heading = -M_PI / 2;
    }

    void update(double dt)
    {
        auto x = m_state.position;
        auto v = m_state.speed;
        auto a = m_state.acceleration;
        auto alpha = m_state.heading;
        auto theta = m_state.wheel_angle;

        if (m_breaking)
        {
            static constexpr double c_breaking_time = 5.0;
            static constexpr double c_acceleration =
                c_max_speed / c_breaking_time;

            if (std::abs(v) > c_epsilon)
            {
                double sign = v > 0 ? -1 : 1;
                v += sign * c_acceleration * dt;
            }
            else
            {
                v = 0.0;
            }
        }
        else
        {
            static constexpr double c_acceleration_time = 0.25;
            static constexpr double c_jerk =
                c_max_acceleration / c_acceleration_time;

            if (m_accelerating || m_accelerating_backwards)
            {
                if (m_accelerating)
                {
                    a += c_jerk * dt;
                }
                else
                {
                    a -= c_jerk * dt;
                }
            }
            else
            {
                if (std::abs(a) > c_epsilon)
                {
                    double sign = a > 0 ? -1 : 1;
                    a += sign * c_jerk * dt;
                }
                else
                {
                    a = 0.0;
                }
            }
        }

        static constexpr double c_steering_time = 0.5;

        if (std::abs(v) > c_epsilon &&
            ((m_turning_left && theta > 0) || (m_turning_right && theta < 0) ||
             (!m_turning_left && !m_turning_right)))
        {
            if (std::abs(theta) > c_epsilon)
            {
                const double sign = theta > 0 ? -1 : 1;
                const double steering_rate =
                    sign * c_max_wheel_angle / c_steering_time * 0.975 *
                    std::pow(std::abs(v) / c_max_speed, 1.0 / 8);

                theta += steering_rate * dt;
            }
            else
            {
                theta = 0.0;
            }
        }
        else if (m_turning_left || m_turning_right)
        {
            const double sign = m_turning_left ? -1 : 1;
            const double steering_rate =
                sign * c_max_wheel_angle / c_steering_time *
                (1.0 - 0.975 * std::pow(std::abs(v) / c_max_speed, 1.0 / 8));

            theta += steering_rate * dt;
        }

        const double w = std::tan(theta) / c_vehicle_wheel_base * v;
        m_w = w;

        if (std::abs(w) <= c_epsilon)
        {
            const Eigen::Rotation2Dd R{alpha};
            m_state.position = x + R * Vector2d{v + a * dt / 2, 0.0} * dt;
        }
        else
        {
            const double delta_x =
                1.0 / w / w *
                ((v * w + a * w * dt) * std::sin(alpha + w * dt) +
                 a * std::cos(alpha + w * dt) - v * w * std::sin(alpha) -
                 a * std::cos(alpha));
            const double delta_y =
                1.0 / w / w *
                ((-v * w - a * w * dt) * std::cos(alpha + w * dt) +
                 a * std::sin(alpha + w * dt) + v * w * std::cos(alpha) -
                 a * std::sin(alpha));

            m_state.position = x + Vector2d{delta_x, delta_y};
        }

        m_state.speed = v + a * dt;

        m_state.speed =
            std::max(-c_max_speed, std::min(m_state.speed, c_max_speed));

        m_state.acceleration = a;
        m_state.acceleration = std::max(
            -c_max_acceleration,
            std::min(m_state.acceleration, c_max_acceleration));

        m_state.heading = alpha + w * dt;
        m_state.heading = m_state.heading;

        m_state.wheel_angle = theta;
        m_state.wheel_angle = std::max(
            -c_max_wheel_angle,
            std::min(m_state.wheel_angle, c_max_wheel_angle));
    }

    VehicleState getState() const
    {
        return m_state;
    }

    double getYawRate() const
    {
        return m_w;
    }

    void setTurningLeft(bool value)
    {
        m_turning_left = value;
    }

    void setTurningRight(bool value)
    {
        m_turning_right = value;
    }

    void setAccelerating(bool value)
    {
        m_accelerating = value;
    }

    void setAcceleratingBackwards(bool value)
    {
        m_accelerating_backwards = value;
    }

    void setBreaking(bool value)
    {
        m_breaking = value;
    }

private: // fields
    VehicleState m_state;
    bool m_turning_left = false;
    bool m_turning_right = false;
    bool m_accelerating = false;
    bool m_accelerating_backwards = false;
    bool m_breaking = false;

    double m_w = 0.0;
};

class CWidget final : public QWidget
{
public: // methods
    explicit CWidget()
    {
        startTimer(0);

        const std::size_t dp = sizeof(UkfState) / sizeof(double);
        const std::size_t mp = sizeof(UkfMeasurement) / sizeof(double);
        const std::size_t cp = sizeof(UkfControl) / sizeof(double);

        m_model_ptr = new CUkfSystemModel{};
        cv::tracking::AugmentedUnscentedKalmanFilterParams params{
            static_cast<int>(dp),
            static_cast<int>(mp),
            static_cast<int>(cp),
            1e-1,
            1e-3,
            m_model_ptr,
            CV_64F};

        UkfState initial_state;
        initial_state.position = Vector2d{0.0, 0.0};
        initial_state.speed = 0.0;
        initial_state.heading = -M_PI / 2;

        params.stateInit = structToCvMat(initial_state);

        m_filter = cv::tracking::createAugmentedUnscentedKalmanFilter(params);
    }

protected: // methods
    void timerEvent(QTimerEvent *) override
    {
        VehicleState state = m_controller.getState();
        double v = state.speed;

        double dt =
            std::chrono::duration_cast<std::chrono::duration<double>>(
                std::chrono::high_resolution_clock::now() - m_last_update)
                .count();
        m_last_update = std::chrono::high_resolution_clock::now();
        m_controller.update(dt);

        state = m_controller.getState();
        double a = (state.speed - v) / dt;

        UkfControl ukf_control;
        UkfMeasurement ukf_measurement;

        ukf_control.acceleration = a;
        ukf_control.yaw_rate = m_controller.getYawRate();

        ukf_measurement.position = state.position;
        ukf_measurement.speed = state.speed;
        ukf_measurement.heading = state.heading;

        m_model_ptr->setDeltaTime(dt);

        m_filter->predict(structToCvMat(ukf_control));
        UkfState ukf_state = cvMatToStruct<UkfState>(
            m_filter->correct(structToCvMat(ukf_measurement)));

        const Eigen::Rotation2Dd R{ukf_state.heading};
        m_ukf_trajectory.emplace_back(
            ukf_state.position + R * Vector2d{c_vehicle_wheel_base / 2, 0.0});
    }

    void keyPressEvent(QKeyEvent *e) override
    {
        switch (e->key())
        {
        case Qt::Key_Escape:
            if (isFullScreen())
            {
                showNormal();
            }
            else
            {
                close();
            }
            break;

        case Qt::Key_F:
        case Qt::Key_F11:
            if (isFullScreen())
            {
                showNormal();
            }
            else
            {
                showFullScreen();
            }
            break;

        case Qt::Key_Q:
            close();
            break;
        }

        if (!e->isAutoRepeat())
        {
            switch (e->key())
            {
            case Qt::Key_Left:
                m_controller.setTurningLeft(true);
                break;

            case Qt::Key_Right:
                m_controller.setTurningRight(true);
                break;

            case Qt::Key_Up:
                m_controller.setAccelerating(true);
                break;

            case Qt::Key_Down:
                m_controller.setAcceleratingBackwards(true);
                break;

            case Qt::Key_Space:
                m_controller.setBreaking(true);
                break;
            }
        }
    }

    void keyReleaseEvent(QKeyEvent *e) override
    {
        if (!e->isAutoRepeat())
        {
            switch (e->key())
            {
            case Qt::Key_Left:
                m_controller.setTurningLeft(false);
                break;

            case Qt::Key_Right:
                m_controller.setTurningRight(false);
                break;

            case Qt::Key_Up:
                m_controller.setAccelerating(false);
                break;

            case Qt::Key_Down:
                m_controller.setAcceleratingBackwards(false);
                break;

            case Qt::Key_Space:
                m_controller.setBreaking(false);
                break;
            }
        }
    }

    void paintEvent(QPaintEvent *) override
    {
        static constexpr double c_car_width = c_vehicle_wheel_base / 1.75;

        VehicleState vehicle_state = m_controller.getState();

        auto x = vehicle_state.position;
        double alpha = vehicle_state.heading;
        const Eigen::Rotation2Dd R{alpha};
        x += R * Vector2d{c_vehicle_wheel_base / 2, 0.0};

        QPointF pos{x.x(), x.y()};

        static constexpr QRgb c_bg_color = qRgb(30, 30, 30);

        QPainter p{this};
        p.fillRect(rect(), c_bg_color);

        p.translate(rect().center());

        static const auto gain = [](double x, double k) {
            double a = 0.5 * pow(2.0 * ((x < 0.5) ? x : 1.0 - x), k);
            return (x < 0.5) ? a : 1.0 - a;
        };

        if (m_camera_pos.isNull())
        {
            m_camera_pos = m_scale * -pos;
        }
        double len =
            QVector2D(m_scale * -pos - m_camera_pos).length() / height();
        double c_cam_alpha = gain(len, 3);
        m_camera_pos =
            c_cam_alpha * m_scale * -pos + (1.0 - c_cam_alpha) * m_camera_pos;

        m_trajectory.emplace_back(pos);

        p.setPen(QPen{Qt::white, 2});

        p.save();
        p.translate(m_camera_pos);

        QPainterPath path;
        if (!m_trajectory.empty())
        {
            path.moveTo(m_scale * m_trajectory[0]);
            for (std::size_t i = 1; i < m_trajectory.size(); i++)
            {
                path.lineTo(m_scale * m_trajectory[i]);
            }
        }
        p.strokePath(path, QPen{Qt::gray, 1});

        path = QPainterPath();
        if (!m_ukf_trajectory.empty())
        {
            path.moveTo(
                m_scale *
                QPointF{m_ukf_trajectory[0].x(), m_ukf_trajectory[0].y()});
            for (std::size_t i = 1; i < m_ukf_trajectory.size(); i++)
            {
                path.lineTo(
                    m_scale *
                    QPointF{m_ukf_trajectory[i].x(), m_ukf_trajectory[i].y()});
            }
        }
        p.strokePath(path, QPen{Qt::green, 1});

        const auto draw_vehicle = [&]() {
            p.drawRect(QRect(
                m_scale * 0,
                m_scale * -c_car_width / 2,
                m_scale * c_vehicle_wheel_base,
                m_scale * c_car_width));
        };

        const auto draw_wheel = [&]() {
            p.fillRect(
                QRect(
                    m_scale * -c_vehicle_wheel_base / 8,
                    m_scale * -c_car_width / 8,
                    m_scale * c_vehicle_wheel_base / 4,
                    m_scale * c_car_width / 4),
                Qt::white);
        };

        QPointF base_pos{vehicle_state.position.x(),
                         vehicle_state.position.y()};

        p.translate(m_scale * base_pos);
        p.rotate(vehicle_state.heading / M_PI * 180);
        draw_vehicle();

        p.save();
        p.translate(m_scale * c_vehicle_wheel_base, m_scale * -c_car_width / 2);
        p.rotate(vehicle_state.wheel_angle / M_PI * 180);
        draw_wheel();
        p.restore();

        p.save();
        p.translate(m_scale * c_vehicle_wheel_base, m_scale * c_car_width / 2);
        p.rotate(vehicle_state.wheel_angle / M_PI * 180);
        draw_wheel();
        p.restore();

        p.save();
        p.translate(m_scale * 0, m_scale * -c_car_width / 2);
        draw_wheel();
        p.restore();

        p.save();
        p.translate(m_scale * 0, m_scale * c_car_width / 2);
        draw_wheel();
        p.restore();

        p.restore();

        QPointF text_pos = QPointF{200, -200};

        QString info;
        info.sprintf(
            "Speed: %.04f km/h\n"
            "Acceleration: %.04f m/s^2\n"
            "Heading: %.04f deg\n"
            "Wheel angle: %.04f deg\n",
            vehicle_state.speed / 3.6,
            vehicle_state.acceleration,
            std::fmod(vehicle_state.heading + 2 * M_PI, 2 * M_PI) / M_PI * 180,
            vehicle_state.wheel_angle / M_PI * 180);

        p.setFont(QFontDatabase::systemFont(QFontDatabase::FixedFont));
        p.drawText(QRect(text_pos.x(), text_pos.y(), 1000, 1000), info);

        QTimer::singleShot(16, this, SLOT(update()));
    }

private: // fields
    CVehicleController m_controller;
    std::vector<QPointF> m_trajectory;
    std::chrono::high_resolution_clock::time_point m_last_update =
        std::chrono::high_resolution_clock::now();
    QPointF m_camera_pos;
    double m_scale = 15;

    cv::Ptr<CUkfSystemModel> m_model_ptr;
    cv::Ptr<cv::tracking::UnscentedKalmanFilter> m_filter;
    std::vector<Vector2d> m_ukf_trajectory;
};

} // namespace

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);

    CWidget widget;
    widget.show();

    return app.exec();
}

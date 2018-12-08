#include <iostream>
#include <Eigen/Geometry>
#include <QtCore/QDebug>
#include <QtCore/QElapsedTimer>
#include <QtCore/QTimer>
#include <QtGui/QFontDatabase>
#include <QtGui/QKeyEvent>
#include <QtGui/QPainter>
#include <QtGui/QPainterPath>
#include <QtWidgets/QApplication>
#include <QtWidgets/QWidget>

namespace {

static constexpr double c_epsilon = 1e-4;
static constexpr double c_wheel_base = 40;
static constexpr double c_max_speed = 5000;
static constexpr double c_max_acceleration = 1e3;
static constexpr double c_max_wheel_angle = M_PI / 5;

struct VehicleState
{
    Eigen::Vector2d position;
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
            static constexpr double c_breaking_time = 1.0;
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

        if (m_turning_left)
        {
            static constexpr double c_steering_time = 0.5;
            const double c_steering_rate = c_max_wheel_angle / c_steering_time *
                                           (1.1 - std::abs(v) / c_max_speed);

            theta -= c_steering_rate * dt;
        }
        else if (m_turning_right)
        {
            static constexpr double c_steering_time = 0.5;
            const double c_steering_rate = c_max_wheel_angle / c_steering_time *
                                           (1.1 - std::abs(v) / c_max_speed);

            theta += c_steering_rate * dt;
        }
        else if (std::abs(theta) > c_epsilon)
        {
            static constexpr double c_steering_time = 0.125;
            const double c_steering_rate =
                c_max_wheel_angle / c_steering_time * std::abs(v) / c_max_speed;

            const double sign = theta > 0 ? -1 : 1;
            theta += sign * c_steering_rate * dt;
        }
        else
        {
            theta = 0.0;
        }

        const double w = std::tan(theta) / c_wheel_base * v;

        if (std::abs(w) <= c_epsilon)
        {
            const Eigen::Rotation2Dd R{alpha};
            m_state.position =
                x + R * Eigen::Vector2d{v * dt + a * dt * dt / 2, 0.0};
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

            m_state.position = x + Eigen::Vector2d{delta_x, delta_y};
        }

        m_state.speed = v + a * dt;

        m_state.speed =
            std::max(-c_max_speed, std::min(m_state.speed, c_max_speed));

        m_state.acceleration = a;
        m_state.acceleration = std::max(
            -c_max_acceleration,
            std::min(m_state.acceleration, c_max_acceleration));

        m_state.heading = alpha + w * dt;
        m_state.heading = std::fmod(m_state.heading + 2 * M_PI, 2 * M_PI);

        m_state.wheel_angle = theta;
        m_state.wheel_angle = std::max(
            -c_max_wheel_angle,
            std::min(m_state.wheel_angle, c_max_wheel_angle));
    }

    VehicleState getState() const
    {
        return m_state;
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
};

class CWidget final : public QWidget
{
public: // methods
    explicit CWidget()
    {
        startTimer(0);
    }

protected: // methods
    void timerEvent(QTimerEvent *) override
    {
        static constexpr double c_nsecs_in_sec = 1e-9;
        double dt = m_timer.nsecsElapsed() * c_nsecs_in_sec;
        m_controller.update(dt);
        m_timer.restart();
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
        static constexpr double c_car_width = c_wheel_base / 1.75;

        VehicleState vehicle_state = m_controller.getState();

        auto x = vehicle_state.position;
        double alpha = vehicle_state.heading;
        const Eigen::Rotation2Dd R{alpha};
        x += R * Eigen::Vector2d{c_wheel_base / 2, 0.0};

        QPointF pos{x.x(), x.y()};

        static constexpr QRgb c_bg_color = qRgb(30, 30, 30);

        QPainter p{this};
        p.fillRect(rect(), c_bg_color);

        p.translate(rect().center());

        static QPointF camera_pos = -pos;
        double c_cam_alpha = std::pow(
            std::atan(QVector2D(-pos - camera_pos).length() / height()) / M_PI *
                2,
            1.5);
        camera_pos = c_cam_alpha * -pos + (1.0 - c_cam_alpha) * camera_pos;

        m_trajectory.emplace_back(pos);

        p.setPen(QPen{Qt::white, 2});

        p.save();
        p.translate(camera_pos);

        QPainterPath path;
        if (!m_trajectory.empty())
        {
            path.moveTo(m_trajectory[0]);
            for (std::size_t i = 1; i < m_trajectory.size(); i++)
            {
                path.lineTo(m_trajectory[i]);
            }
        }
        p.strokePath(path, QPen{Qt::gray, 1});

        QPointF base_pos{vehicle_state.position.x(),
                         vehicle_state.position.y()};
        p.translate(base_pos);
        p.rotate(vehicle_state.heading / M_PI * 180);
        p.drawRect(QRect(0, -c_car_width / 2, c_wheel_base, c_car_width));
        p.save();
        p.translate(c_wheel_base, -c_car_width / 2);
        p.rotate(vehicle_state.wheel_angle / M_PI * 180);
        p.fillRect(
            QRect(
                -c_wheel_base / 8,
                -c_car_width / 8,
                c_wheel_base / 4,
                c_car_width / 4),
            Qt::white);
        p.restore();
        p.save();
        p.translate(c_wheel_base, c_car_width / 2);
        p.rotate(vehicle_state.wheel_angle / M_PI * 180);
        p.fillRect(
            QRect(
                -c_wheel_base / 8,
                -c_car_width / 8,
                c_wheel_base / 4,
                c_car_width / 4),
            Qt::white);
        p.restore();
        p.save();
        p.translate(0, -c_car_width / 2);
        p.fillRect(
            QRect(
                -c_wheel_base / 8,
                -c_car_width / 8,
                c_wheel_base / 4,
                c_car_width / 4),
            Qt::white);
        p.restore();
        p.save();
        p.translate(0, c_car_width / 2);
        p.fillRect(
            QRect(
                -c_wheel_base / 8,
                -c_car_width / 8,
                c_wheel_base / 4,
                c_car_width / 4),
            Qt::white);
        p.restore();
        p.restore();

        QPointF text_pos = QPointF{200, -200};

        QString info;
        info.sprintf(
            "Speed: %.04f pix/s\n"
            "Acceleration: %.04f pix/s^2\n"
            "Heading: %.04f deg\n"
            "Wheel angle: %.04f deg\n",
            vehicle_state.speed,
            vehicle_state.acceleration,
            vehicle_state.heading / M_PI * 180,
            vehicle_state.wheel_angle / M_PI * 180);

        p.setFont(QFontDatabase::systemFont(QFontDatabase::FixedFont));
        p.drawText(QRect(text_pos.x(), text_pos.y(), 1000, 1000), info);

        QTimer::singleShot(16, this, SLOT(update()));
    }

private: // fields
    QElapsedTimer m_timer;
    CVehicleController m_controller;
    std::vector<QPointF> m_trajectory;
};

} // namespace

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);

    CWidget widget;
    widget.show();

    return app.exec();
}

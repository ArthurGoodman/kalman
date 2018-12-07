#include <iostream>
#include <Eigen/Geometry>
#include <QtCore/QDebug>
#include <QtCore/QElapsedTimer>
#include <QtCore/QTimer>
#include <QtGui/QKeyEvent>
#include <QtGui/QPainter>
#include <QtGui/QPainterPath>
#include <QtWidgets/QApplication>
#include <QtWidgets/QWidget>
#include "Types.hpp"

namespace {

class CVehicleController final
{
public: // methods
    explicit CVehicleController()
    {
    }

    void update(double dt)
    {
        Eigen::Vector2d a{0.0, 0.0};

        if (m_up_button_pressed)
        {
            a -= Eigen::Vector2d{0, 1} * 1e4;
        }

        if (m_down_button_pressed)
        {
            a += Eigen::Vector2d{0, 1} * 1e4;
        }

        double w = 0.0;

        if (m_left_button_pressed)
        {
            w -= 1e3;
        }

        if (m_right_button_pressed)
        {
            w += 1e3;
        }

        auto position = m_state.getPosition();
        auto velocity = m_state.getVelocity();
        double rotation = m_state.getRotation().x();
        const Eigen::Rotation2Dd R{rotation * M_PI / 180};

        m_state.setPosition(position + R * velocity * dt);
        m_state.setVelocity((velocity + a * dt) * (1 - 1 * dt));
        m_state.setRotation(Vector1d{rotation + w * dt});
    }

    CVehicleState getState() const
    {
        return m_state;
    }

    void setLeftButtonState(bool pressed)
    {
        m_left_button_pressed = pressed;
    }

    void setRightButtonState(bool pressed)
    {
        m_right_button_pressed = pressed;
    }

    void setUpButtonState(bool pressed)
    {
        m_up_button_pressed = pressed;
    }

    void setDownButtonState(bool pressed)
    {
        m_down_button_pressed = pressed;
    }

private: // fields
    CVehicleState m_state;
    bool m_left_button_pressed = false;
    bool m_right_button_pressed = false;
    bool m_up_button_pressed = false;
    bool m_down_button_pressed = false;
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
        double dt = m_timer.nsecsElapsed() * 1e-9;
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
                m_controller.setLeftButtonState(true);
                break;

            case Qt::Key_Right:
                m_controller.setRightButtonState(true);
                break;

            case Qt::Key_Up:
                m_controller.setUpButtonState(true);
                break;

            case Qt::Key_Down:
                m_controller.setDownButtonState(true);
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
                m_controller.setLeftButtonState(false);
                break;

            case Qt::Key_Right:
                m_controller.setRightButtonState(false);
                break;

            case Qt::Key_Up:
                m_controller.setUpButtonState(false);
                break;

            case Qt::Key_Down:
                m_controller.setDownButtonState(false);
                break;
            }
        }
    }

    void paintEvent(QPaintEvent *) override
    {
        static constexpr double c_scale = 2.0;

        CVehicleState vehicle_state = m_controller.getState();

        QPointF pos{vehicle_state.getPosition().x(),
                    vehicle_state.getPosition().y()};

        static constexpr QRgb c_bg_color = qRgb(30, 30, 30);

        QPainter p{this};
        p.fillRect(rect(), c_bg_color);

        p.translate(rect().center() - pos);

        m_trajectory.emplace_back(pos);

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

        p.translate(pos);
        p.rotate(vehicle_state.getRotation().x());
        p.scale(c_scale, c_scale);

        p.setPen(Qt::white);
        p.drawRect(QRect(-5, -10, 10, 20));

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

#pragma once

#include <cstddef>
#include <ostream>
#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>

using Vector1d = Eigen::Matrix<double, 1, 1, false>;

template <std::size_t Size>
class CDataVectorBase
{
public: // fields
    static constexpr std::size_t DataVectorSize = Size;

public: // methods
    explicit CDataVectorBase()
        : m_data(DataVectorSize, 1, CV_64F)
    {
        m_data.setTo(cv::Scalar(0.0));
    }

    explicit CDataVectorBase(const CDataVectorBase &other)
    {
        *this = other;
    }

    explicit CDataVectorBase(CDataVectorBase &&other)
    {
        *this = std::move(other);
    }

    explicit CDataVectorBase(const cv::Mat &mat)
        : m_data(mat)
    {
    }

    CDataVectorBase &operator=(const CDataVectorBase &other)
    {
        m_data = other.m_data.clone();
        return *this;
    }

    CDataVectorBase &operator=(CDataVectorBase &&other)
    {
        m_data = std::move(other.m_data);
        return *this;
    }

    cv::Mat getData() const
    {
        return m_data;
    }

    friend std::ostream &operator<<(
        std::ostream &stream,
        const CDataVectorBase &data)
    {
        cv::Mat transposed;
        cv::transpose(data.m_data, transposed);
        stream << transposed;
        return stream;
    }

    friend bool operator==(
        const CDataVectorBase &lhs,
        const CDataVectorBase &rhs)
    {
        return cv::countNonZero(lhs.m_data != rhs.m_data) == 0;
    }

protected: // types
    template <std::size_t N>
    using VectorType = Eigen::Matrix<double, static_cast<int>(N), 1, false>;

protected: // methods
    template <std::size_t N, std::size_t Offset>
    VectorType<N> getProperty() const
    {
        static_assert(Offset + N <= DataVectorSize, "");

        VectorType<N> value;

        for (std::size_t i = 0; i < N; i++)
        {
            value(i) = m_data.at<double>(static_cast<int>(Offset + i));
        }

        return value;
    }

    template <std::size_t N, std::size_t Offset>
    void setProperty(const VectorType<N> &value)
    {
        static_assert(Offset + N <= DataVectorSize, "");

        for (std::size_t i = 0; i < N; i++)
        {
            m_data.at<double>(static_cast<int>(Offset + i)) = value(i);
        }
    }

private: // fields
    cv::Mat m_data;
};

#define BEGIN_DATA_VECTOR_TYPE(Name, Size)            \
    class Name final : public ::CDataVectorBase<Size> \
    {                                                 \
    public:                                           \
        using CDataVectorBase::CDataVectorBase;

#define END_DATA_VECTOR_TYPE \
    }                        \
    ;

#define DATA_VECTOR_PROPERTY(Name, Size, Offset)                   \
    CDataVectorBase::VectorType<Size> get##Name() const            \
    {                                                              \
        return getProperty<Size, Offset>();                        \
    }                                                              \
    void set##Name(const CDataVectorBase::VectorType<Size> &value) \
    {                                                              \
        setProperty<Size, Offset>(value);                          \
    }

BEGIN_DATA_VECTOR_TYPE(CVehicleState, 5)
DATA_VECTOR_PROPERTY(Position, 2, 0)
DATA_VECTOR_PROPERTY(Velocity, 2, 2)
DATA_VECTOR_PROPERTY(Rotation, 1, 4)
END_DATA_VECTOR_TYPE

// Derived from UGPM -- Unified Gaussian Preintegrated Measurements
// https://github.com/CedricLeGentil/ugpm
// Copyright 2021 Cedric Le Gentil, released under the MIT License.
//
// Modified for Go-RIO: the preintegration consumes radar Doppler ego-velocity
// in place of accelerometer measurements, so the translation is integrated once
// rather than twice.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef PREINT_TYPES_H
#define PREINT_TYPES_H

#include <iostream>
#include <algorithm>
#include <memory>
#include <numeric>
#include "Eigen/Dense"


namespace ugpm
{

    // Preintegration methods
    enum PreintType {LPM, UGPM};

    inline PreintType strToPreintType(std::string type)
    {
        std::transform(type.begin(), type.end(), type.begin(), [](unsigned char c){ return std::tolower(c); });
        PreintType output;
        if(type == "lpm") output = LPM;
        else if(type == "ugpm") output = UGPM;
        else
        {
            throw std::range_error("The type of preintegration method is unknown, program stopping now");
        }
        return output;
    };


    typedef Eigen::Matrix<double, 2, 1> Vec2;
    typedef Eigen::Matrix<double, 3, 1> Vec3;
    typedef Eigen::Matrix<double, 6, 1> Vec6;
    typedef Eigen::Matrix<double, 9, 1> Vec9;
    typedef Eigen::Matrix<double, 9, 1> Vec12;
    typedef Eigen::Matrix<double, 18, 1> Vec18;
    typedef Eigen::Matrix<double, Eigen::Dynamic, 1> VecX;
    typedef Eigen::Matrix<double, 1, Eigen::Dynamic> RowX;

    typedef Eigen::Matrix<double, 1, 9> Row9;

    typedef Eigen::Matrix<double, 3, 3> Mat3;
    typedef Eigen::Matrix<double, 6, 6> Mat6;
    typedef Eigen::Matrix<double, 9, 9> Mat9;
    typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> MatX;

    typedef Eigen::Matrix<double, 3, 2> Mat3_2;
    typedef Eigen::Matrix<double, 3, 6> Mat3_6;
    typedef Eigen::Matrix<double, 3, 9> Mat3_9;

    typedef Eigen::Matrix<double, 6, 3> Mat6_3;
    typedef Eigen::Matrix<double, 6, 6> Mat6_6;

    typedef Eigen::Matrix<double, 9, 3> Mat9_3;
    typedef Eigen::Matrix<double, 9, 6> Mat9_6;

    inline Vec3 stdToVec3(const std::vector<double>& v)
    {
        Vec3 output;
        output << v[0], v[1], v[2];
        return output;
    }


    // Data structure to store individual measurements (can be acc or gyr data)
    struct DataSample
    {
        double t;
        double data[3];
    };

    // Data structure to store a collection of IMU measurements
    struct GyroVelData
    {
        // Offset in second for every sample timestamps
        double t_offset = 0.0;

        // Vectors of samples
        std::vector<DataSample> vel;
        std::vector<DataSample> gyr;

        // Noise specification. The velocity is per axis: the radar's elevation
        // spread is 5-7 degrees against a far wider azimuth coverage, so the
        // Doppler least squares determines the vertical component 6-9x less
        // precisely than the horizontal ones on every NTU sequence. Collapsing
        // that to one number told the GP the three axes were equally certain.
        Eigen::Vector3d vel_var;
        double gyr_var;

        GyroVelData(){};

        GyroVelData(const GyroVelData& imu_data):
            t_offset(imu_data.t_offset),
            vel(imu_data.vel),
            gyr(imu_data.gyr),
            vel_var(imu_data.vel_var),
            gyr_var(imu_data.gyr_var)
        {
        };

        bool checkFrequency()
        {
            bool output = true;
            // Get min and max delta t for both the vel and gyr data
            double min_vel_dt = 1e10;
            double max_vel_dt = -1e10;
            for(int i = 0; i < vel.size()-1; ++i)
            {
                double dt = vel[i+1].t - vel[i].t;
                if(dt < min_vel_dt) min_vel_dt = dt;
                if(dt > max_vel_dt) max_vel_dt = dt;
            }
            double min_gyr_dt = 1e10;
            double max_gyr_dt = -1e10;
            for(int i = 0; i < gyr.size()-1; ++i)
            {
                double dt = gyr[i+1].t - gyr[i].t;
                if(dt < min_gyr_dt) min_gyr_dt = dt;
                if(dt > max_gyr_dt) max_gyr_dt = dt;
            }

            if(min_vel_dt < 0)
            {
                std::cout << "WARNING: Velocity data is not sorted in time" << std::endl;
                output = false;
            }
            if(min_gyr_dt < 0)
            {
                std::cout << "WARNING: Gyroscope data is not sorted in time" << std::endl;
                output = false;
            }
            double range_vel_dt = max_vel_dt - min_vel_dt;
            double range_gyr_dt = max_gyr_dt - min_gyr_dt;
            // if ( (range_vel_dt/max_vel_dt > 0.1) || (range_gyr_dt/max_gyr_dt > 0.1) )
            // {
            //     std::cout << "WARNING: Accelerometer or gyroscope data is not sampled at a constant frequency" << std::endl;
            //     std::cout << "\tMin \\ max delta time for acc " << min_acc_dt << " \\ " << max_acc_dt << std::endl;
            //     std::cout << "\tMin \\ max delta time for gyr " << min_gyr_dt << " \\ " << max_gyr_dt << std::endl;
            //     output = false;
            // }
            return output;
        }
        // Return the collection of samples in between the two timestamps
        GyroVelData get(double from, double to)
        {
            // Check if the query interval makes sense
            std::setprecision(15);
            // std::cout << "From = " << from << " to = " << to << std::endl;
            if(from <= to)
            {
                GyroVelData output;

                output.t_offset = t_offset;
                output.vel_var = vel_var;
                output.gyr_var = gyr_var;
                output.vel = get(vel, from, to);
                output.gyr = get(gyr, from, to);
                return output;
            }
            // If the query inteval does not make sense throw an exception
            else
            {
                throw std::invalid_argument("The argument of GyroVelData::Get are not consistent");
            }
        }

        void print()
        {
            std::cout << "Imu data with offset of " << t_offset << std::endl;
            std::cout << "  Velocity data (" << vel.size() << " samples with std = "
                      << vel_var.cwiseSqrt().transpose() << ")" << std::endl;
            for(const auto& a : vel)
            {
                std::cout << "      t = " << a.t << ":    "
                        << a.data[0] << "    "
                        << a.data[1] << "    "
                        << a.data[2] << std::endl;
            }
            std::cout << "  Gyroscope data (" << gyr.size() << " samples with std = " << sqrt(gyr_var) << ")" << std::endl;
            for(const auto& g : gyr)
            {
                std::cout << "      t = " << g.t << ":    " 
                        << g.data[0] << "    "
                        << g.data[1] << "    "
                        << g.data[2] << std::endl;
            }
        }

        private:
            // Return the collection of samples in between the two timestamps
            std::vector<DataSample> get(const std::vector<DataSample>& samples, double from, double to)
            {
                std::vector<DataSample> output;
                if(from >= to)
                {
                    std::cout << "WARNING: Trying to get IMU data from a time range null or negative" << std::endl;
                    return output;
                }

                // TODO: could be optimised (keeping an internal pointer for repetitive calls instead for looping through all the data)
                int i = 0;
                bool loop = true;
                while(loop)
                {
                    if(samples[i].t > from)
                    {
                        if(samples[i].t < to)
                        {
                            output.push_back(samples[i]);
                        }
                        else
                        {
                            loop = false;
                        }
                    }
                    if(i < (samples.size() - 1) )
                    {
                        i++;
                    }
                    else
                    {
                        loop = false;
                    }
                }

                return output;
            }
    };
    typedef std::shared_ptr<GyroVelData> GyroVelDataPtr;


    // Structure to store the different elements of a preintegrated measurement
    struct PreintMeasBasic{
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        Mat3 delta_R;
        Vec3 delta_p;
        // Vec3 delta_pint;
        double dt; 
        double dt_sq_half;

        void print() const
        {
            std::cout << "Preintegrated measurement" << std::endl;
            std::cout << "  dt = " << dt << std::endl;
            std::cout << "  dt_sq_half = " << dt_sq_half << std::endl;
            std::cout << "  Delta R = " << std::endl
                    << "    " << delta_R.row(0) << std::endl
                    << "    " << delta_R.row(1) << std::endl
                    << "    " << delta_R.row(2) << std::endl;
            std::cout << "  Delta p = " << std::endl 
                    << "    " << delta_p.row(0) << std::endl
                    << "    " << delta_p.row(1) << std::endl
                    << "    " << delta_p.row(2) << std::endl;
            // std::cout << "  Delta pint = " << std::endl
            //         << "    " << delta_pint.row(0) << std::endl
            //         << "    " << delta_pint.row(1) << std::endl
            //         << "    " << delta_pint.row(2) << std::endl;
        }
    };

    struct PreintMeas : PreintMeasBasic
    {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        // Mat9 cov;
        Mat6 cov;

        Mat3 d_delta_R_d_bw;
        Vec3 d_delta_R_d_t;
     
        Mat3 d_delta_p_d_bw;
        Mat3 d_delta_p_d_bv;
        Vec3 d_delta_p_d_t;
     
        // Mat3 d_delta_pint_d_bw;
        // Mat3 d_delta_pint_d_bv;
        // Vec3 d_delta_pint_d_t;

        PreintMeas(Mat3 d_R,
                Vec3 d_p,
                double dt, 
                double dt_sq_half,
                // Mat9 cov_mat) :
                Mat6 cov_mat) :
                PreintMeasBasic{d_R, d_p, dt, dt_sq_half}, cov(cov_mat) {}; 

        PreintMeas(){};


        // void printAll() const
        // {
        //     print();
        //     std::cout << "  Covariance = " << std::endl
        //             << "    " << cov.row(0) << std::endl
        //             << "    " << cov.row(1) << std::endl
        //             << "    " << cov.row(2) << std::endl
        //             << "    " << cov.row(3) << std::endl
        //             << "    " << cov.row(4) << std::endl
        //             << "    " << cov.row(5) << std::endl
        //             << "    " << cov.row(6) << std::endl
        //             << "    " << cov.row(7) << std::endl
        //             << "    " << cov.row(8) << std::endl;
        //     std::cout << "  Jacobian Delta R / gyr bias = " << std::endl
        //             << "    " << d_delta_R_d_bw.row(0) << std::endl
        //             << "    " << d_delta_R_d_bw.row(1) << std::endl
        //             << "    " << d_delta_R_d_bw.row(2) << std::endl;
        //     std::cout << "  Jacobian Delta R / time-shift = " << std::endl
        //             << "    " << d_delta_R_d_t.transpose() << std::endl;
        //     std::cout << "  Jacobian Delta v / gyr bias = " << std::endl
        //             << "    " << d_delta_p_d_bw.row(0) << std::endl
        //             << "    " << d_delta_p_d_bw.row(1) << std::endl
        //             << "    " << d_delta_p_d_bw.row(2) << std::endl;
        //     std::cout << "  Jacobian Delta v / acc bias = " << std::endl
        //             << "    " << d_delta_p_d_bv.row(0) << std::endl
        //             << "    " << d_delta_p_d_bv.row(1) << std::endl
        //             << "    " << d_delta_p_d_bv.row(2) << std::endl;
        //     std::cout << "  Jacobian Delta v / time-shift = " << std::endl
        //             << "    " << d_delta_p_d_t.transpose() << std::endl;
        //     std::cout << "  Jacobian Delta p / gyr bias = " << std::endl
        //             << "    " << d_delta_pint_d_bw.row(0) << std::endl
        //             << "    " << d_delta_pint_d_bw.row(1) << std::endl
        //             << "    " << d_delta_pint_d_bw.row(2) << std::endl;
        //     std::cout << "  Jacobian Delta p / acc bias = " << std::endl
        //             << "    " << d_delta_pint_d_bv.row(0) << std::endl
        //             << "    " << d_delta_pint_d_bv.row(1) << std::endl
        //             << "    " << d_delta_pint_d_bv.row(2) << std::endl;
        //     std::cout << "  Jacobian Delta p / time-shift = " << std::endl
        //             << "    " << d_delta_pint_d_t.transpose() << std::endl;
        // }


    };
    typedef std::shared_ptr<PreintMeas> PreintMeasPtr;


    struct PreintOption
    {
        double min_freq = 500;
        PreintType type = UGPM;
        double quantum = -1;
        double state_freq = 50.0;
        bool correlate = true;
    };

    struct PreintPrior
    {
        std::vector<double> vel_bias = {0,0,0};
        std::vector<double> gyr_bias = {0,0,0};
    };

    
    struct GPSeHyper
    {
        // Lengthscale squared
        double l2;
        // Sigma squared of the signal (in front of exponential)
        double sf2;
        // Sigma squared of measurement noise
        double sz2;
        // Mean of signal
        double mean;
    };


     // Function to sort indexes
    template <typename T>
    std::vector<int> sortIndexes(const std::vector<T> &v)
    {

        // initialize original index locations
        std::vector<int> idx(v.size());
        std::iota(idx.begin(), idx.end(), 0); 

        // sort indexes based on comparing values in v
        std::sort(idx.begin(), idx.end(),
            [&v](int i1, int i2) {return v[i1] < v[i2];});

        return idx;
    }


    // Tool to sort vector of vector and keep track of original indexes
    template <class T>
    class SortIndexTracker2
    {

        public:
            SortIndexTracker2( std::vector<std::vector<T> >& data):
                data_(&data)
            {
                std::vector<T> temp_data;
                std::vector<std::pair<int, int> > temp_map;
                for(int i = 0; i < data_->size(); ++i)
                {
                    temp_data.insert(temp_data.end(), data_->at(i).begin(), data_->at(i).end());
                    for(int j = 0; j < data_->at(i).size(); ++j)
                    {
                        temp_map.push_back(std::make_pair(i,j));
                    }
                }
                
                std::vector<int> sorted_indexes = sortIndexes(temp_data);

                for(int i = 0; i < sorted_indexes.size(); ++i)
                {
                    index_map_.push_back(temp_map[sorted_indexes[i]]);
                }
            }

            T get(const int i) const
            {
                return (*data_)[index_map_[i].first][index_map_[i].second];
            }

            std::pair<int, int> getIndexPair(const int i) const
            {
                return index_map_[i];
            }

            std::vector<int> getIndexVector(const int vec_i) const
            {
                std::vector<int> output;
                for(int i = 0; i < index_map_.size(); ++i)
                {
                    if(index_map_[i].first == vec_i) output.push_back(i);
                }
                return output;
            }

            template <class U>
            std::vector<U> getVector(const std::vector<U>& data, const int vec_i) const
            {
                std::vector<U> output;
                for(int i = 0; i < index_map_.size(); ++i)
                {
                    if(index_map_[i].first == vec_i) output.push_back(data[i]);
                }
                return output;
            }


            // Get the 1D index corresponding to the 2D indices
            int getIndex(const int i, const int j) const
            {
                for(int k = 0; k < index_map_.size(); ++k)
                {
                    if((index_map_[k].first == i) && (index_map_[k].second == j))
                    {
                        return k;
                    }
                }
                return -1;
            }

            // Get from the 2D indices in a 1D structure
            template <class U>
            U get(const int i, const int j, const std::vector<U>& data) const
            {
                for(int k = 0; k < index_map_.size(); ++k)
                {
                    if((index_map_[k].first == i) && (index_map_[k].second == j))
                    {
                        return data[k];
                    }
                }
                throw std::range_error("SortIndexTracker2: trying to 'get' with indices not present in the index_map");
            }


            // Get from the 2D indices in a 1D structure
            template <class U>
            std::vector<U> applySort(const std::vector<std::vector<U> >& data) const
            {
                std::vector<U> output;
                for(int k = 0; k < index_map_.size(); ++k)
                {
                    output.push_back(data[index_map_[k].first][index_map_[k].second]);
                }
                return output;
            }


            T back() const
            {
                return (*data_)[index_map_.back().first][index_map_.back().second];
            }

            int size() const
            {
                return index_map_.size();
            }
            
            T getSmallestGap()
            {
                T diff = get(1) - get(0);
                for(int i = 1; i < (size()-1); ++i)
                {
                    diff = get(i+1) - get(i);
                }
                return diff;
            }

        private:
            std::vector<std::vector<T> >* data_;

            std::vector<std::pair<int, int> > index_map_;
            

    };

}

#endif

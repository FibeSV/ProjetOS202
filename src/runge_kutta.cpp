#include <iostream>
#include "runge_kutta.hpp"
#include "cartesian_grid_of_speed.hpp"
using namespace Geometry;

Geometry::CloudOfPoints
Numeric::solve_RK4_fixed_vortices(double dt, CartesianGridOfSpeed const &t_velocity, Geometry::CloudOfPoints const &t_points)
{
    constexpr double onesixth = 1. / 6.;
    using vector = Simulation::Vortices::vector;
    using point = Simulation::Vortices::point;

    Geometry::CloudOfPoints newCloud(t_points.numberOfPoints());
    // On ne bouge que les points :
    for (std::size_t iPoint = 0; iPoint < t_points.numberOfPoints(); ++iPoint)
    {
        point p = t_points[iPoint];
        vector v1 = t_velocity.computeVelocityFor(p);
        point p1 = p + 0.5 * dt * v1;
        p1 = t_velocity.updatePosition(p1);
        vector v2 = t_velocity.computeVelocityFor(p1);
        point p2 = p + 0.5 * dt * v2;
        p2 = t_velocity.updatePosition(p2);
        vector v3 = t_velocity.computeVelocityFor(p2);
        point p3 = p + dt * v3;
        p3 = t_velocity.updatePosition(p3);
        vector v4 = t_velocity.computeVelocityFor(p3);
        newCloud[iPoint] = t_velocity.updatePosition(p + onesixth * dt * (v1 + 2. * v2 + 2. * v3 + v4));
    }
    return newCloud;
}

Geometry::CloudOfPoints
Numeric::solve_RK4_movable_vortices(double dt, CartesianGridOfSpeed &t_velocity,
                                    Simulation::Vortices &t_vortices,
                                    Geometry::CloudOfPoints const &t_points)
{
    MPI_Comm global;
    int rank, nbp;

    MPI_Comm_dup(MPI_COMM_WORLD, &global);
    MPI_Comm_size(global, &nbp);
    MPI_Comm_rank(global, &rank);
    std::cout << "Comm size: " << nbp << "Current rank: " << rank << std::endl;
    int np_cloud_cal = nbp - 2;                                              // the number of processes to calculate the cloud calculation
    std::size_t devided_nm_p = t_points.numberOfPoints() / np_cloud_cal + 1; // the number of points to be calculated in a single process

    MPI_Datatype MPI_Point;

    int len[2] = {1, 1};
    MPI_Aint pos[2] = {offsetof(Geometry::Point<double>, x), offsetof(Geometry::Point<double>, y)};
    MPI_Datatype typ[2] = {MPI_DOUBLE, MPI_DOUBLE};

    MPI_Type_create_struct(2, len, pos, typ, &MPI_Point);
    MPI_Type_commit(&MPI_Point);

    constexpr double onesixth = 1. / 6.;
    using vector = Simulation::Vortices::vector;
    using point = Simulation::Vortices::point;

    //---------------Vortex Center Calculation-------------------
    if (rank == 0)
    {
        std::vector<point> newVortexCenter;
        newVortexCenter.reserve(t_vortices.numberOfVortices());

        auto start = std::chrono::system_clock::now();

        // Second loop
        // #pragma omp parallel for
        for (std::size_t iVortex = 0; iVortex < t_vortices.numberOfVortices(); ++iVortex)
        {
            point p = t_vortices.getCenter(iVortex);
            vector v1 = t_vortices.computeSpeed(p);
            point p1 = p + 0.5 * dt * v1;
            p1 = t_velocity.updatePosition(p1);
            vector v2 = t_vortices.computeSpeed(p1);
            point p2 = p + 0.5 * dt * v2;
            p2 = t_velocity.updatePosition(p2);
            vector v3 = t_vortices.computeSpeed(p2);
            point p3 = p + dt * v3;
            p3 = t_velocity.updatePosition(p3);
            vector v4 = t_vortices.computeSpeed(p3);
            newVortexCenter.emplace_back(t_velocity.updatePosition(p + onesixth * dt * (v1 + 2. * v2 + 2. * v3 + v4)));
        }

        auto end = std::chrono::system_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        // std::cout << "time spent in the 2nd loop: " << duration.count() << std::endl;

        // Third Loop
        start = std::chrono::system_clock::now();

        // #pragma omp parallel for
        for (std::size_t iVortex = 0; iVortex < t_vortices.numberOfVortices(); ++iVortex)
            t_vortices.setVortex(iVortex, newVortexCenter[iVortex], t_vortices.getIntensity(iVortex));

        end = std::chrono::system_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        // std::cout << "time spent in the 3rd loop: " << duration.count() << std::endl << std::endl;

        t_velocity.updateVelocityField(t_vortices);
    }

    //---------------Merge the results and return----------------
    else if (rank == 1)
    {
        // Collect the sub cloud data 
        Geometry::CloudOfPoints *new_sub_clouds = new Geometry::CloudOfPoints[np_cloud_cal];
        for (int i = 0; i < np_cloud_cal; i++)
        {
            new_sub_clouds[i] = Geometry::CloudOfPoints(std::min((i + 1) * devided_nm_p, t_points.numberOfPoints()) - i * devided_nm_p);
            MPI_Recv(new_sub_clouds[i].data(), new_sub_clouds[i].numberOfPoints(), MPI_Point, i + 2, i + 2, global, new MPI_Status());
        }

        //Merge the sub clouds together
        Geometry::CloudOfPoints newCloud;
        std::vector<Point<double>> newCloud_temp;
        newCloud_temp.reserve(t_points.numberOfPoints());
        for (int i = 0; i < np_cloud_cal; i++)
        {
            newCloud_temp.insert(newCloud_temp.end(), new_sub_clouds[i].begin(), new_sub_clouds[i].end());
        }

        for (int i = 0; i < int(newCloud_temp.size()); i++)
        {
            newCloud.addAPoint(newCloud_temp[i]);
        }
        
        return newCloud;
    }

    //---------------------Cloud Calculation---------------------
    else
    {

        Geometry::CloudOfPoints new_sub_cloud(std::min((rank - 1) * devided_nm_p, t_points.numberOfPoints()) - (rank - 2) * devided_nm_p);

        // On ne bouge que les points :
        // First Loop
        auto start = std::chrono::system_clock::now();

#pragma omp parallel for
        for (std::size_t iPoint = (rank - 2) * devided_nm_p;
             iPoint < std::min((rank - 1) * devided_nm_p, t_points.numberOfPoints());
             ++iPoint)
        {
            point p = t_points[iPoint];
            vector v1 = t_velocity.computeVelocityFor(p);
            point p1 = p + 0.5 * dt * v1;
            p1 = t_velocity.updatePosition(p1);
            vector v2 = t_velocity.computeVelocityFor(p1);
            point p2 = p + 0.5 * dt * v2;
            p2 = t_velocity.updatePosition(p2);
            vector v3 = t_velocity.computeVelocityFor(p2);
            point p3 = p + dt * v3;
            p3 = t_velocity.updatePosition(p3);
            vector v4 = t_velocity.computeVelocityFor(p3);
            new_sub_cloud[iPoint] = t_velocity.updatePosition(p + onesixth * dt * (v1 + 2. * v2 + 2. * v3 + v4));
        }
        auto end = std::chrono::system_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        // std::cout << "time spent in the 1st loop: " << duration.count() << std::endl;
        MPI_Send(new_sub_cloud.data(),
                 std::min((rank - 1) * devided_nm_p, t_points.numberOfPoints()) - (rank - 2) * devided_nm_p,
                 MPI_Point, 1, rank, global);
    }
}
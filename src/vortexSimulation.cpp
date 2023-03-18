#include <SFML/Window/Keyboard.hpp>
#include <ios>
#include <iostream>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <tuple>
#include <chrono>
#include "cartesian_grid_of_speed.hpp"
#include "vortex.hpp"
#include "cloud_of_points.hpp"
#include "runge_kutta.hpp"
#include "screen.hpp"
#include <mpi.h>
auto readConfigFile( std::ifstream& input )
{
    using point=Simulation::Vortices::point;

    int isMobile;
    std::size_t nbVortices;
    Numeric::CartesianGridOfSpeed cartesianGrid;
    Geometry::CloudOfPoints cloudOfPoints;
    constexpr std::size_t maxBuffer = 8192;
    char buffer[maxBuffer];
    std::string sbuffer;
    std::stringstream ibuffer;
    // Lit la première ligne de commentaire :
    input.getline(buffer, maxBuffer); // Relit un commentaire
    input.getline(buffer, maxBuffer);// Lecture de la grille cartésienne
    sbuffer = std::string(buffer,maxBuffer);
    ibuffer = std::stringstream(sbuffer);
    double xleft, ybot, h;
    std::size_t nx, ny;
    ibuffer >> xleft >> ybot >> nx >> ny >> h;
    cartesianGrid = Numeric::CartesianGridOfSpeed({nx,ny}, point{xleft,ybot}, h);
    input.getline(buffer, maxBuffer); // Relit un commentaire
    input.getline(buffer, maxBuffer); // Lit mode de génération des particules
    sbuffer = std::string(buffer,maxBuffer);
    ibuffer = std::stringstream(sbuffer);
    int modeGeneration;
    ibuffer >> modeGeneration;
    if (modeGeneration == 0) // Génération sur toute la grille 
    {
        std::size_t nbPoints;
        ibuffer >> nbPoints;
        cloudOfPoints = Geometry::generatePointsIn(nbPoints, {cartesianGrid.getLeftBottomVertex(), cartesianGrid.getRightTopVertex()});
    }
    else 
    {
        std::size_t nbPoints;
        double xl, xr, yb, yt;
        ibuffer >> xl >> yb >> xr >> yt >> nbPoints;
        cloudOfPoints = Geometry::generatePointsIn(nbPoints, {point{xl,yb}, point{xr,yt}});
    }
    // Lit le nombre de vortex :
    input.getline(buffer, maxBuffer); // Relit un commentaire
    input.getline(buffer, maxBuffer); // Lit le nombre de vortex
    sbuffer = std::string(buffer, maxBuffer);
    ibuffer = std::stringstream(sbuffer);
    try {
        ibuffer >> nbVortices;        
    } catch(std::ios_base::failure& err)
    {
        std::cout << "Error " << err.what() << " found" << std::endl;
        std::cout << "Read line : " << sbuffer << std::endl;
        throw err;
    }
    Simulation::Vortices vortices(nbVortices, {cartesianGrid.getLeftBottomVertex(),
                                               cartesianGrid.getRightTopVertex()});
    input.getline(buffer, maxBuffer);// Relit un commentaire
    for (std::size_t iVortex=0; iVortex<nbVortices; ++iVortex)
    {
        input.getline(buffer, maxBuffer);
        double x,y,force;
        std::string sbuffer(buffer, maxBuffer);
        std::stringstream ibuffer(sbuffer);
        ibuffer >> x >> y >> force;
        vortices.setVortex(iVortex, point{x,y}, force);
    }
    input.getline(buffer, maxBuffer);// Relit un commentaire
    input.getline(buffer, maxBuffer);// Lit le mode de déplacement des vortex
    sbuffer = std::string(buffer,maxBuffer);
    ibuffer = std::stringstream(sbuffer);
    ibuffer >> isMobile;
    return std::make_tuple(vortices, isMobile, cartesianGrid, cloudOfPoints);
}


int main( int nargs, char* argv[] )
{
    MPI_Comm global;
    int rank, nbp;

    MPI_Init(&nargs, &argv);
    MPI_Comm_dup(MPI_COMM_WORLD, &global);
    MPI_Comm_size(global, &nbp);
    MPI_Comm_rank(global, &rank);

    std::cout << nbp << ' ' << rank << std::endl;
    
    char const* filename;
    if (nargs==1)
    {
        std::cout << "Usage : vortexsimulator <nom fichier configuration>" << std::endl;
        return EXIT_FAILURE;
    }

    filename = argv[1];
    std::ifstream fich(filename);
    auto config = readConfigFile(fich);
    fich.close();



    std::size_t resx=800, resy=600;
    if (nargs>3)
    {
        resx = std::stoull(argv[2]);
        resy = std::stoull(argv[3]);
    }

    auto vortices = std::get<0>(config);
    auto isMobile = std::get<1>(config);
    auto grid     = std::get<2>(config);
    auto cloud    = std::get<3>(config);
    
    grid.updateVelocityField(vortices);
    
    bool animate=false;
    double dt = 0.1;
    bool advance = false;
    
    int tag = 404;
    bool screen_open = true;

    if (rank==0){
        std::cout << "######## Vortex simultor ########" << std::endl << std::endl;
        std::cout << "Press P for play animation " << std::endl;
        std::cout << "Press S to stop animation" << std::endl;
        std::cout << "Press right cursor to advance step by step in time" << std::endl;
        std::cout << "Press down cursor to halve the time step" << std::endl;
        std::cout << "Press up cursor to double the time step" << std::endl;
        Graphisme::Screen myScreen( {resx,resy}, {grid.getLeftBottomVertex(), grid.getRightTopVertex()} );
        
        while (myScreen.isOpen()){
            auto start = std::chrono::system_clock::now();
            sf::Event event;
            while (myScreen.pollEvent(event))
            {
                // évènement "fermeture demandée" : on ferme la fenêtre
                if (event.type == sf::Event::Closed)
                    myScreen.close();
                if (event.type == sf::Event::Resized)
                {
                    // on met à jour la vue, avec la nouvelle taille de la fenêtre
                    myScreen.resize(event);
                }
                if (sf::Keyboard::isKeyPressed(sf::Keyboard::P)) animate = true;
                if (sf::Keyboard::isKeyPressed(sf::Keyboard::S)) animate = false;
                if (sf::Keyboard::isKeyPressed(sf::Keyboard::Up)) dt *= 2;
                if (sf::Keyboard::isKeyPressed(sf::Keyboard::Down)) dt /= 2;
                if (sf::Keyboard::isKeyPressed(sf::Keyboard::Right)) advance = true;
                MPI_Bcast(&animate, 1, MPI_BYTE, 0, global);
                MPI_Bcast(&advance, 1, MPI_BYTE, 0, global);
                MPI_Bcast(&dt, 1, MPI_FLOAT, 0, global);
            }
            std::cout << "hello"<<std::endl;
            MPI_Status status;
            //std::vector<double>  m_coordinates(cloud.numberOfPoints());
            MPI_Recv(cloud.data(), cloud.numberOfPoints(), MPI_FLOAT, 1, MPI_ANY_TAG, global, &status);
            std::cout << cloud.data()[0] << std::endl;
            //Geometry::CloudOfPoints cloud(m_coordinates);
            myScreen.clear(sf::Color::Black);
            std::string strDt = std::string("Time step : ") + std::to_string(dt);
            myScreen.drawText(strDt, Geometry::Point<double>{50, double(myScreen.getGeometry().second-96)});
            myScreen.displayVelocityField(grid, vortices);
            myScreen.displayParticles(grid, vortices, cloud);
            auto end = std::chrono::system_clock::now();
            std::chrono::duration<double> diff = end - start;
            std::string str_fps = std::string("FPS : ") + std::to_string(1./diff.count());
            myScreen.drawText(str_fps, Geometry::Point<double>{300, double(myScreen.getGeometry().second-96)});
            myScreen.display();
        }
        screen_open = myScreen.isOpen();
        MPI_Bcast(&screen_open, 1, MPI_BYTE, 0, global);

    }
    
    if (rank==1){

        while (screen_open)
        {
            //     MPI_Bcast(&screen_open, 1, MPI_BYTE, 0, global);
            // // on inspecte tous les évènements de la fenêtre qui ont été émis depuis la précédente itération
            //     MPI_Bcast(&animate, 1, MPI_BYTE, 0, global);
            //     MPI_Bcast(&advance, 1, MPI_BYTE, 0, global);
            //     MPI_Bcast(&dt, 1, MPI_FLOAT, 0, global);
                if (animate | advance)
                {
                    if (isMobile)
                    {
                        cloud = Numeric::solve_RK4_movable_vortices(dt, grid, vortices, cloud);
                    }
                    else
                    {
                        cloud = Numeric::solve_RK4_fixed_vortices(dt, grid, cloud);
                    }
                    std::cout << "sleep slepp" <<std::endl;
                    MPI_Send(cloud.data(), cloud.numberOfPoints(), MPI_FLOAT, 1, tag, global);

                }

        }
    }
    return EXIT_SUCCESS;
 }
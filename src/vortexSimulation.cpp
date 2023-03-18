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
#include <thread>
#include <mpi.h>

int CALCULATION_RESULT_TAG = 888;
int CALCULATION_REQUEST_TAG = 88;
int INTERFACE_RANK = 0;
int CALCULATION_RANK = 1;

auto readConfigFile( std::ifstream& input ) {
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
    // MPI initialization
    MPI_Comm global;    // MPI communicator
    int rank, nbp;      // MPI rank and number of processes

    MPI_Init(&nargs, &argv);
    MPI_Comm_dup(MPI_COMM_WORLD, &global);  // Duplicate the communicator to get a local communicator
    MPI_Comm_size(global, &nbp);            // Get the total number of processes
    MPI_Comm_rank(global, &rank);           // Get the rank of the current process

    MPI_Datatype MPI_Point;                 // Declare an MPI datatype for Point object

    int          len[2] = { 1, 1 };         // Size of each element of the struct in the datatype
    MPI_Aint     pos[2] = { offsetof(Geometry::Point<double>, x), offsetof(Geometry::Point<double>, y)};  // Offset of each element of the struct in the datatype
    MPI_Datatype typ[2] = { MPI_DOUBLE, MPI_DOUBLE };  // Type of each element of the struct in the datatype

    // Create the MPI datatype for Point object
    MPI_Type_create_struct( 2, len, pos, typ, &MPI_Point );
    MPI_Type_commit( &MPI_Point );

    char const* filename;
    if (nargs==1)
    {
        std::cout << "Usage : vortexsimulator <nom fichier configuration>" << std::endl;
        return EXIT_FAILURE;
    }

    filename = argv[1];
    std::ifstream fich(filename);
    auto config = readConfigFile(fich);     // Read configuration from file
    fich.close();

    std::size_t resx=800, resy=600;
    if (nargs>3)
    {
        resx = std::stoull(argv[2]);        // Get the resolution of the screen from command line argument
        resy = std::stoull(argv[3]);
    }

    auto vortices = std::get<0>(config);   // Get the list of vortices from configuration
    auto isMobile = std::get<1>(config);   // Get whether the vortices are mobile or not from configuration
    auto grid     = std::get<2>(config);   // Get the grid information from configuration
    auto cloud    = std::get<3>(config);   // Get the cloud information from configuration
    
    grid.updateVelocityField(vortices);     // Update the velocity field of the grid using the vortices

    double dt = 0.1;
    if (rank == INTERFACE_RANK){
        // Code for the interface process
        std::cout << "######## Vortex simultor ########" << std::endl << std::endl;
        std::cout << "Press P for play animation " << std::endl;
        std::cout << "Press S to stop animation" << std::endl;
        std::cout << "Press right cursor to advance step by step in time" << std::endl;
        std::cout << "Press down cursor to halve the time step" << std::endl;
        std::cout << "Press up cursor to double the time step" << std::endl;
        
        Graphisme::Screen myScreen( {resx,resy}, {grid.getLeftBottomVertex(), grid.getRightTopVertex()} );
        bool animate = false;

        while (myScreen.isOpen()){
            auto start = std::chrono::system_clock::now();
            bool advance = false;

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

            sf::Event event;
            do {
                while (myScreen.pollEvent(event)) {
                    // évènement "fermeture demandée" : on ferme la fenêtre
                    if (event.type == sf::Event::Closed) {
                        myScreen.close();
                        dt = 0;
                    }
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
                }
            } while (!animate && !advance && myScreen.isOpen());

            //SEND CALCULATION REQUEST
            MPI_Send(&dt, 1, MPI_DOUBLE, CALCULATION_RANK, CALCULATION_REQUEST_TAG, global);

            //RECEIVE CALCULATION RESULT
            MPI_Recv(cloud.data(), cloud.numberOfPoints(), MPI_Point, CALCULATION_RANK, CALCULATION_RESULT_TAG, global, new MPI_Status());
            MPI_Recv(grid.data(), grid.cellGeometry().first*grid.cellGeometry().second, MPI_DOUBLE, CALCULATION_RANK, CALCULATION_RESULT_TAG+1, global, new MPI_Status());
            MPI_Recv(vortices.data(),  vortices.numberOfVortices()*3, MPI_DOUBLE, CALCULATION_RANK, CALCULATION_RESULT_TAG+2, global, new MPI_Status());
        }
    }
    
    if (rank == CALCULATION_RANK){
        do {
            if (isMobile) {
                cloud = Numeric::solve_RK4_movable_vortices(dt, grid, vortices, cloud);
            } else {
                cloud = Numeric::solve_RK4_fixed_vortices(dt, grid, cloud);
            }
            //SEND CALCULATION RESULT

            MPI_Send(cloud.data(), cloud.numberOfPoints(), MPI_Point, INTERFACE_RANK, CALCULATION_RESULT_TAG, global);
            MPI_Send(grid.data(),  grid.cellGeometry().first*grid.cellGeometry().second, MPI_DOUBLE, INTERFACE_RANK, CALCULATION_RESULT_TAG+1, global);
            MPI_Send(vortices.data(),  vortices.numberOfVortices()*3, MPI_DOUBLE, INTERFACE_RANK, CALCULATION_RESULT_TAG+2, global);
            //RECEIVE CALCULATION REQUEST
            MPI_Recv( &dt, 1, MPI_DOUBLE, INTERFACE_RANK, CALCULATION_REQUEST_TAG, global, new MPI_Status() );
        } while (dt > 0);
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}

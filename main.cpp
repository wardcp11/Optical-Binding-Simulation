#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>
#ifdef _DEBUG
#ifndef DBG_NEW
#define DBG_NEW new ( _NORMAL_BLOCK , __FILE__ , __LINE__ )
#define new DBG_NEW
#endif
#endif  // _DEBUG


#include <iostream>
using namespace std;
#include <math.h>
#include <complex>
#include <random>

#include <chrono>
using namespace std::chrono;



//DISABLE SANITIZE IN VISUAL STUDIO PROPERTIES WHEN DONE    https://learn.microsoft.com/en-us/cpp/sanitizers/asan?view=msvc-170
//Some Visual Studio Options Need to be disabled to sanitize dont forget to reenable

//POTENTIAL PROBLEMS
    //CASE INSIDE GG FUNCTION WHEN N=M
    //COULD BE USING WRONG VALUES FOR X,Y,Z MAY NEED TO CHANGE FROM POSITION TO SOME OTHER VALUE



//Math Constants
double pi = 3.14159265358979323846;
const complex<double> i(0.0, 1.0);
double xx[3] = { 1.0, 0.0, 0.0 };
double yy[3] = { 0.0, 1.0, 0.0 };
double zz[3] = { 0.0, 0.0, 1.0 };

//UNIT HELL
double microsecond = 1.0;
double microgram = 1.0;
double nanometer = 1.0;
double femtocoulomb = 1.0;
double kelvin = 1.0;
double second = microsecond * 1e6;
double kg = microgram * 1e9;
double meter = nanometer * 1e9;
double coulomb = femtocoulomb * 1e15;
double joule = kg * pow(meter, 2.0) * pow(second, -2.0);
double volt = joule / coulomb;
double farad = coulomb / volt;
double ampere = coulomb / second;
double watt = joule / second;
double henry = pow(second, 2) / farad;

//Fundemental Constants
double epsilon_0 = 8.854187817e-12 * farad / meter;
double mu_0 = pi * (4e-7) * henry / meter; //
double n_0 = sqrt(mu_0 / epsilon_0);
double c = 1.0 / (sqrt(epsilon_0 * mu_0));
double kB = 1.38e-23 * kg * pow(meter, 2) / (pow(second, 2) * kelvin);

//Colloid Properties
double a = 500e-9 * meter;
double VOL = pi * pow(a, 3.0) * (4.0 / 3.0);
double rhoSIO2 = 2320 * kg / pow(meter, 3);
double mass = VOL * rhoSIO2;
double mu = 1.6e-3 * kg / (meter * second); //dynamic viscosity
double gamma = 6.0 * pi * mu * a / mass; //drag frequency
int T = 300;

//Laser Properties
double lambda = 600e-9 * meter;
double area = 100 * pow(1e-6 * meter, 2);
double power = 1000.0 * watt;
double I_0 = power / area;

//Polarizability
double epsilon_p = -3 * 1.33 * 1.33; // for SiO2
double mu_p = 1.0;
double epsilon_b = 1.33 * 1.33; //  for SiO2
double mu_b = 1.0;
double n_b = sqrt((mu_b) / (epsilon_b));
double k0 = 2.0 * pi / lambda;
double omega = k0 * c;
double k = k0 * sqrt(epsilon_b);
complex<double> alpha_sr = 4.0 * pi * epsilon_0 * epsilon_b * a * a * a * ((epsilon_p - epsilon_b) / (epsilon_p + 2 * epsilon_b));
complex<double> as = 1.0 / (1.0 / alpha_sr + -1.0 * i * (k * k * k) / (6 * pi * epsilon_0 * epsilon_b));

//Physical Setup
double L = 25e-6 * meter; //Computational Area
int NN = 2; //Number of Particles
complex<double> alpha = alpha_sr.real() + i * alpha_sr.imag();

//Incident Field
double w_0 = L / 4;
double E_0 = sqrt(2 * n_0 * n_b * I_0);

//Time Evolution
double dt = 10 / gamma;
double Gamma = 2.0 * gamma * kB * T / mass;
double DeltaB = Gamma * dt;
int maxstep = 500;

bool zeroZAxis = true; //Special Modifier to Deal with Undefined Behavior

bool RREF(complex<double>** A, int equations);

complex<double>* EInc(double x, double y, double z) { //returns incident field at specified x,y,z in xx direction
    //RETURNS HEAP ARRAY AND NEEDS TO BE FREED AFTER USED
    char eIncDirection = 'x'; //Define Char for direction 'x', 'y', or 'z'      defaults to x in undefined behavior
    complex<double>* incidentField = new complex<double>[3];
    incidentField[0] = 0.0;
    incidentField[1] = 0.0;
    incidentField[2] = 0.0;
    switch (eIncDirection) {
    case 'x':
        incidentField[0] = E_0 * exp(i * k * z) * exp(-1 * (x * x + y * y) / (w_0 * w_0));
        break;
    case 'y':
        incidentField[1] = E_0 * exp(i * k * z) * exp(-1 * (x * x + y * y) / (w_0 * w_0));
        break;
    case 'z':
        incidentField[2] = E_0 * exp(i * k * z) * exp(-1 * (x * x + y * y) / (w_0 * w_0));
        break;
    default:
        incidentField[0] = E_0 * exp(i * k * z) * exp(-1 * (x * x + y * y) / (w_0 * w_0));
        cout << "Noncritical Error: Incident Field Axis Not Defined, defaulting to X" << endl;
    }

    return incidentField;
}

complex<double>** GreensFunc(double x, double y, double z) { //returns greens function at specified x,y,z
    //RETURNS HEAP ARRAY AND NEEDS TO BE FREED AFTER USED
    //indexed G[y][x]

    complex<double>** G = new complex<double>*[3];
    for (int y = 0; y < 3; ++y) {
        G[y] = new complex<double>[3];
    }

    //reused variables
    double r = sqrt(x * x + y * y + z * z);
    complex<double> ikr = i * k * r;
    complex<double> coeff = exp(ikr) / (4.0 * pi * r * r * r * r * r * epsilon_0 * epsilon_b); //same coefficent for each element (includes denominator)
    complex<double> offDiag = coeff * -1.0 * (-3.0 + 3.0 * ikr + k * k * r * r); //all off diagonal elements are the same times variables

    G[0][0] = coeff * ((y * y + z * z) * (-1 + k * k * (y * y + z * z) + ikr) + x * x * (2 + k * k * (y * y + z * z) - 2.0 * ikr));
    G[0][1] = offDiag * x * y;
    G[0][2] = offDiag * x * z;

    G[1][0] = offDiag * x * y;
    G[1][1] = coeff * k * k * x * x * x * x + z * z * (-1.0 + k * k * z * z + ikr) + y * y * (2.0 + k * k * z * z - 2.0 * ikr) + x * x * (-1.0 + ikr + k * k * (y * y + 2.0 * z * z));
    G[1][2] = offDiag * y * z;

    G[2][0] = offDiag * x * z;
    G[2][1] = offDiag * y * z;
    G[2][2] = coeff * k * k * x * x * x * x + k * k * y * y * y * y + 2.0 * z * z * (1.0 - ikr) + y * y * (-1.0 + k * k * z * z + ikr) + x * x * (-1.0 + ikr + k * k * (2.0 * y * y + z * z));

    return G;
}

complex<double>** GG(double*** positions, int step, int particle_N, int particle_M) { //greens function from particle N on particle M
    //RETURNS HEAP ARRAY AND NEEDS TO BE FREED AFTER USED
    //CAREFUL WITH THIS FUNCTION, I BELIEVE WHEN IT IS CALLED IT ALLOCATES MEMORY FOR BOTH GreensFunc and GG

    if (particle_M != particle_N) {
        double MtoN_x = positions[step][particle_N][0] - positions[step][particle_M][0];
        double MtoN_y = positions[step][particle_N][1] - positions[step][particle_M][1];
        double MtoN_z = positions[step][particle_N][2] - positions[step][particle_M][2];
        return GreensFunc(MtoN_x, MtoN_y, MtoN_z);
    }
    else {
        complex<double>** G = new complex<double>*[3];
        for (int y = 0; y < 3; ++y) {
            G[y] = new complex<double>[3];
        }

        G[0][0] = -1.0 / (alpha * positions[step][particle_N][0]);
        G[0][1] = 0.0;
        G[0][2] = 0.0;

        G[1][0] = 0.0;
        G[1][1] = -1.0 / (alpha * positions[step][particle_N][1]);
        G[1][2] = 0.0;

        G[2][0] = 0.0;
        G[2][1] = 0.0;
        G[2][2] = -1.0 / (alpha * positions[step][particle_N][2]);

        return G;
    }
}

complex<double>** SolveForP_n(double*** positions, int step) { //Linear equation solver for dipole moments
    //RETURNS HEAP ARRAY AND NEEDS TO BE FREED AFTER USED
    //uses RREF to solve linear equations for dipole moments
    //solution is an array in RREF with elements x_1, y_1, z_1, ... , x_n, y_n, z_n

    complex<double>** rrefArray = new complex<double>*[3 * NN]; // 3 * #particles because 3 axes
    for (int equation = 0; equation < 3 * NN; ++equation) {
        rrefArray[equation] = new complex<double>[3 * NN + 1];
    }

    //calculated for each particle each time, definitely could be optimized
    for (int particle_m = 0; particle_m < NN; ++particle_m) {

        complex<double>* incidentField = EInc(positions[step][particle_m][0], positions[step][particle_m][1], positions[step][particle_m][2]);

        rrefArray[0 + 3 * particle_m][3 * NN] = -1.0 * incidentField[0];
        rrefArray[1 + 3 * particle_m][3 * NN] = -1.0 * incidentField[1];
        rrefArray[2 + 3 * particle_m][3 * NN] = -1.0 * incidentField[2];

        for (int particle_n = 0; particle_n < NN; ++particle_n) {
            complex<double>** G = GG(positions, step, particle_n, particle_m);

            rrefArray[0 + 3 * particle_m][0 + 3 * particle_n] = G[0][0];
            rrefArray[0 + 3 * particle_m][1 + 3 * particle_n] = G[0][1];
            rrefArray[0 + 3 * particle_m][2 + 3 * particle_n] = G[0][2];

            rrefArray[1 + 3 * particle_m][0 + 3 * particle_n] = G[1][0];
            rrefArray[1 + 3 * particle_m][1 + 3 * particle_n] = G[1][1];
            rrefArray[1 + 3 * particle_m][2 + 3 * particle_n] = G[1][2];

            rrefArray[2 + 3 * particle_m][0 + 3 * particle_n] = G[2][0];
            rrefArray[2 + 3 * particle_m][1 + 3 * particle_n] = G[2][1];
            
            //Dealing With Undefined Behavior when m=n and z=0, G[2][2] is infinity
            if (zeroZAxis == false) {
                rrefArray[2 + 3 * particle_m][2 + 3 * particle_n] = G[2][2];
            }
            else{
                if (isinf( real( G[2][2] ) ) == true) { // known that dipole moment z component will be zero
                    rrefArray[2 + 3 * particle_m][2 + 3 * particle_n] = 1.0;
                }
                else {
                    rrefArray[2 + 3 * particle_m][2 + 3 * particle_n] = 0.0;
                }
            }

            for (int y = 0; y < 3; ++y) { delete[] G[y]; }
            delete[] G;

        }
        delete[] incidentField;
    }

    RREF(rrefArray, 3 * NN); //MAGIC

    return rrefArray;
}



int main() {

    auto start = high_resolution_clock::now();

    // *** START OF PROGRAM SETUP ***
    // 
    //  Reserve Memory
    //      3d positions
    //      2d velocities
    //      2d polarizabilities
    //      2d Incident Electric Field

    //Initialize Positions with Random Locations
    std::random_device rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> dis(-(L / 2), L / 2);
    double*** positions = new double**[maxstep]; // indexed positions[time step][particle#][axis] (time step: 0-maxstep) (axis: x=0, y=1, z=2)
    for (int step = 0; step < maxstep; ++step) {
        positions[step] = new double* [NN];
        for (int particle = 0; particle < NN; ++particle) {
            positions[step][particle] = new double[3];
        }
    }
    for (int particle = 0; particle < NN; ++particle) {
        for (int axis = 0; axis < 3; ++axis) { // initialize for each axis x,y,z
            if (zeroZAxis == true){
                if (axis != 2) {//random positions for particles
                    positions[0][particle][axis] = dis(rd);
                }
                else {//set z position element to 0
                    positions[0][particle][axis] = 0.0;
                }
            }
            else {
                positions[0][particle][axis] = dis(rd);
            }
        }
    }
    //Initialize Velocities with 0, can change
    double** velocities = new double*[NN];
    for (int particle = 0; particle < NN; ++particle) {
        velocities[particle] = new double[3];
        for (int axis = 0; axis < 3; ++axis) { // initialize for each axis x,y,z
            velocities[particle][axis] = 0;
        }
    }
    //Initialize Polarizabilities
    complex<double>** polarizabilities = new complex<double>*[NN];
    for (int particle = 0; particle < NN; ++particle) {
        polarizabilities[particle] = new complex<double>[3];
        for (int axis = 0; axis < 3; ++axis) { // initialize for each axis x,y,z
            polarizabilities[particle][axis] = alpha;
        }
    }
    //Initialize Electric Field
    complex<double>** incElectricField = new complex<double>* [NN];
    for (int particle = 0; particle < NN; ++particle) {
        incElectricField[particle] = EInc(positions[0][particle][0], positions[0][particle][1], positions[0][particle][2]);
    }




    complex<double>** temp = SolveForP_n(positions, 0); //Temp Code to print off RREF array
    for (int i = 0; i < 3 * NN; ++i) {
        for (int j = 0; j < 3 * NN + 1; ++j) {
            cout << temp[i][j];
        }
        cout << endl;
    }
    for (int particle = 0; particle < 3 * NN; ++particle) { delete[] temp[particle]; } // deallocate each "sub" array
    delete[] temp; // delete overall array





    // *** START OF SIMULATION ***

    for (int step = 1; step < maxstep; ++step) {
        for (int particle = 0; particle < NN; ++particle) {
            for (int axis = 0; axis < 3; ++axis) {
                positions[step][particle][axis] = positions[step - 1][particle][axis] + velocities[particle][axis] * dt;

            }
        }
    }



    // *** END OF PROGRAM CLEANUP ***
    // 
    //  Free Memory
    //All Variables to be freed
    //      3d positions
    //      2d velocities
    //      2d polarizabilities
    //      2d Incident Electric Field

    for (int step = 0; step < maxstep; ++step){
        for (int particle = 0; particle < NN; ++particle) {
            delete[] positions[step][particle]; // Delete each axis
        }
        delete[] positions[step]; // Delete each particle
    }
    delete[] positions; // delete overall array

    for (int index_1 = 0; index_1 < NN; ++index_1) { delete[] velocities[index_1]; } // deallocate each "sub" array
    delete[] velocities; // delete overall array

    for (int index_1 = 0; index_1 < NN; ++index_1) { delete[] polarizabilities[index_1]; } // deallocate each "sub" array
    delete[] polarizabilities; // delete overall array

    for (int particle = 0; particle < NN; ++particle) { delete[] incElectricField[particle]; } // Free memory for each incident field
    delete[] incElectricField; // delete overall array


    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    cout << "Time[us]: " << duration.count() << endl;

    _CrtSetReportMode(_CRT_WARN, _CRTDBG_MODE_DEBUG);
    _CrtDumpMemoryLeaks();

    return 0;
}


//Helper Functions

bool RREF(complex<double>** A, int equations) {
    //Gauss Jordan Elimination
    //Could Cause Issues Down the Line
    
    for (int i = 0; i < equations; i++) {

        // Find the pivot row (largest absolute value element in column i)
        int pivot_row = i;
        for (int j = i + 1; j < equations; j++) {
            if (abs(A[j][i]) > abs(A[pivot_row][i])) {
                pivot_row = j;
            }
        }

        // Swap the pivot row with the current row i
        for (int j = i; j <= equations; j++) {
            swap(A[i][j], A[pivot_row][j]);
        }

        // If the pivot element is too small, the system has no unique solution
        if (abs(A[i][i]) < 1e-10) {
            cerr << "Matrix is singular or nearly singular." << endl;
            return false; // No solution or infinite solutions
        }

        // Normalize the pivot row by dividing it by the pivot element
        complex<double> pivot = A[i][i];
        for (int j = i; j <= equations; j++) {
            A[i][j] /= pivot;
        }

        // Eliminate all other entries in column i
        for (int j = 0; j < equations; j++) {
            if (j != i) {
                complex<double> factor = A[j][i];
                for (int k = i; k <= equations; k++) {
                    A[j][k] -= factor * A[i][k];
                }
            }
        }
    }
    return true;
}
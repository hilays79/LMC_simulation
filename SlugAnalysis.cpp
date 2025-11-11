#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <tuple>
#include <stdexcept>
#include <H5Cpp.h>
#include <sstream>
#include <iomanip>
#include <H5public.h>
#include <fstream>
#include <filesystem>
#include <slug_cluster.H>
#include <slug_wrapper.h>
#include <slug_predefined.H>
#include <slug_PDF.H>

// Function to read data from an HDF5 file
std::vector<std::pair<std::string, std::vector<double>>> readHDF5(const std::string& filename, const std::string& groupname) {
    std::vector<std::pair<std::string, std::vector<double>>> data;
    try {
        std::cout << "Opening file: " << filename << std::endl;
        H5::H5File file(filename, H5F_ACC_RDONLY);
        std::cout << "File opened successfully." << std::endl;

        std::cout << "Opening group: " << groupname << std::endl;
        H5::Group group = file.openGroup(groupname);
        std::cout << "Group opened successfully." << std::endl;

        std::cout << "Datasets in the group:" << std::endl;
        for (hsize_t i = 0; i < group.getNumObjs(); i++) {
            std::string objName = group.getObjnameByIdx(i);
            H5G_obj_t objType = group.getObjTypeByIdx(i);
            
            if (objType == H5G_DATASET) {
                std::cout << " - " << objName << std::endl;
                
                H5::DataSet dataset = group.openDataSet(objName);
                H5::DataSpace dataspace = dataset.getSpace();
                int ndims = dataspace.getSimpleExtentNdims();
                std::vector<hsize_t> dims(ndims);
                dataspace.getSimpleExtentDims(dims.data(), NULL);

                hsize_t totalSize = 1;
                for (hsize_t dim : dims) {
                    totalSize *= dim;
                }
                // print the total size of the dataset
                std::cout << "Total size: " << totalSize << std::endl;

                std::vector<double> datasetData(totalSize);
                dataset.read(datasetData.data(), H5::PredType::NATIVE_DOUBLE);
                
                data.push_back({objName, std::move(datasetData)});
            }
        }

        file.close();
    } catch(H5::Exception& error) {
        std::cerr << "An error occurred." << std::endl;
        error.printErrorStack();
    }
    return data;
}

slug_cluster_state_noyields createSlugClusterState(const std::vector<std::pair<std::string, std::vector<double>>>& hdf5Data, double current_simulation_time, size_t particleIndex) {
    slug_cluster_state_noyields state;

    // Find the SlugStateRng dataset
    auto it_rng = std::find_if(hdf5Data.begin(), hdf5Data.end(),
                           [](const auto& pair) { return pair.first == "SlugStateRng"; });

    if (it_rng != hdf5Data.end()) {
        const std::vector<double>& slugStateRng = it_rng->second;
        
        if (slugStateRng.size() > particleIndex * 2 + 1) {
            uint64_t part1 = static_cast<uint64_t>(slugStateRng[particleIndex * 2]);
            uint64_t part2 = static_cast<uint64_t>(slugStateRng[particleIndex * 2 + 1]);
            
            // Combine part1 and part2 according to the specified format
            state.rngStateAtBirth = part1 + (static_cast<rng_state_t>(part2) << 64);
        } else {
            std::cerr << "Error: SlugStateRng dataset does not contain data for particle index " << particleIndex << std::endl;
        }
    } else {
        std::cerr << "Error: SlugStateRng dataset not found." << std::endl;
    }

    // Find the RealAge dataset
    auto it_age = std::find_if(hdf5Data.begin(), hdf5Data.end(),
                           [](const auto& pair) { return pair.first == "RealAge"; });
    // Assign it to the state.ClusterAge object
    if (it_age != hdf5Data.end()) {
        const std::vector<double>& realAge = it_age->second;
        
        if (realAge.size() > particleIndex) {
            state.clusterAge = realAge[particleIndex];
            // std::cout << "Maximum value of the real age: " << *std::max_element(realAge.begin(), realAge.end()) << std::endl;
        } else {
            std::cerr << "Error: RealAge dataset does not contain data for particle index " << particleIndex << std::endl;
        }
    } else {
        std::cerr << "Error: RealAge dataset not found." << std::endl;
    }


    // Find the SlugStateInt dataset
    auto it_int = std::find_if(hdf5Data.begin(), hdf5Data.end(),
                           [](const auto& pair) { return pair.first == "SlugStateInt"; });

    if (it_int != hdf5Data.end()) {
        const std::vector<double>& slugStateInt = it_int->second;
        
        if (slugStateInt.size() > particleIndex * 2 + 1) {
            state.id = static_cast<uint64_t>(slugStateInt[particleIndex * 2]);
            state.stoch_sn = static_cast<uint64_t>(slugStateInt[particleIndex * 2 + 1]);
        } else {
            std::cerr << "Error: SlugStateInt dataset does not contain data for particle index " << particleIndex << std::endl;
        }
    } else {
        std::cerr << "Error: SlugStateInt dataset not found." << std::endl;
    }

    // Find the SlugStateDouble dataset
    auto it_double = std::find_if(hdf5Data.begin(), hdf5Data.end(),
                           [](const auto& pair) { return pair.first == "SlugStateDouble"; });

    if (it_double != hdf5Data.end()) {
        const std::vector<double>& slugStateDouble = it_double->second;
        
        if (slugStateDouble.size() >= (particleIndex + 1) * 23) {
            size_t offset = particleIndex * 23;
            state.targetMass = slugStateDouble[offset + 0];
            state.birthMass = slugStateDouble[offset + 1];
            state.aliveMass = slugStateDouble[offset + 2];
            state.stochBirthMass = slugStateDouble[offset + 3];
            state.stochAliveMass = slugStateDouble[offset + 4];
            state.stochRemnantMass = slugStateDouble[offset + 5];
            state.nonStochBirthMass = slugStateDouble[offset + 6];
            state.nonStochAliveMass = slugStateDouble[offset + 7];
            state.nonStochRemnantMass = slugStateDouble[offset + 8];
            state.stellarMass = slugStateDouble[offset + 9];
            state.stochStellarMass = slugStateDouble[offset + 10];
            state.nonStochStellarMass = slugStateDouble[offset + 11];
            state.formationTime = slugStateDouble[offset + 12];
            state.curTime = slugStateDouble[offset + 13];
            // state.clusterAge = slugStateDouble[offset + 14];
            state.lifetime = slugStateDouble[offset + 15];
            state.stellarDeathMass = slugStateDouble[offset + 16];
            state.A_V = slugStateDouble[offset + 17];
            state.A_Vneb = slugStateDouble[offset + 18];
            state.Lbol = slugStateDouble[offset + 19];
            state.Lbol_ext = slugStateDouble[offset + 20];
            state.tot_sn = slugStateDouble[offset + 21];
            state.last_yield_time = slugStateDouble[offset + 22];
        } else {
            std::cerr << "Error: SlugStateDouble dataset does not contain data for particle index " << particleIndex << std::endl;
        }
    } else {
        std::cerr << "Error: SlugStateDouble dataset not found." << std::endl;
    }

    return state;
}

std::vector<double> createSlugWrappersAndGetQH0(const std::vector<std::pair<std::string, std::vector<double>>>& hdf5Data, double current_simulation_time) {
    std::vector<double> ionizingLuminosities;
    
    // Initialize slug_globals if it hasn't been initialized yet
    if (slugWrapper::slug_globals == nullptr) {
        slugWrapper::slug_globals = new slug_predefined();
    }

    auto it_double = std::find_if(hdf5Data.begin(), hdf5Data.end(),
                                  [](const auto& pair) { return pair.first == "SlugStateDouble"; });

    if (it_double != hdf5Data.end()) {
        const std::vector<double>& slugStateDouble = it_double->second;
        size_t numParticles = slugStateDouble.size() / 23;

        for (size_t i = 0; i < numParticles; ++i) {
            // Create a state object for each particle
            slug_cluster_state_noyields state = createSlugClusterState(hdf5Data, current_simulation_time, i);

            // Create a slugWrapper object for each particle
            slugWrapper wrapper(state);

            // Get the ionizing luminosity
            wrapper.advanceToTime(state.clusterAge);
            // print
            // std::cout << "Wrapper advanced in time";
            double ionizingLuminosity = wrapper.getPhotometryQH0();
            ionizingLuminosities.push_back(ionizingLuminosity);

        }
    } else {
        std::cerr << "Error: SlugStateDouble dataset not found." << std::endl;
    }

    return ionizingLuminosities;
}

double calculateMedian(std::vector<double> vec) {
    if (vec.empty()) {
        return 0.0;  // or throw an exception
    }

    size_t n = vec.size() / 2;
    std::nth_element(vec.begin(), vec.begin() + n, vec.end());

    if (vec.size() % 2 == 0) {
        // If even number of elements, return average of middle two
        auto max_it = std::max_element(vec.begin(), vec.begin() + n);
        return (*max_it + vec[n]) / 2.0;
    } else {
        // If odd number of elements, return middle element
        return vec[n];
    }
}

std::tuple<std::string, std::string, int> CreateSnapshotPath(int argc, char* argv[]) {
    std::string key;
    std::string resolution;
    int scenario = 0, snapshot = 0;
    std::ostringstream pathStream;
    std::ostringstream outpathStream;

    if (argc == 3) {
        // argv[1] = MW, argv[2] = snapshot
        std::string label = argv[1];
        try {
            snapshot = std::stoi(argv[2]);
        } catch (const std::invalid_argument& e) {
            std::cerr << "Error: Snapshot must be an integer." << std::endl;
            return std::make_tuple("", "", -1);
        }
        pathStream << "/scratch/jh2/hs9158/results/" << label << "_run_89Msun/snapshot_"
                   << std::setfill('0') << std::setw(3) << snapshot << ".hdf5";
        outpathStream << "/scratch/jh2/hs9158/results/" << label << "_run_89Msun";
        return std::make_tuple(pathStream.str(), outpathStream.str(), snapshot);
    }
    else if (argc == 4){
        resolution = argv[1];
        key = "";
    }
    else if (argc == 5){
        resolution = argv[1];
        key = argv[4];
    }
    else {
        std::cerr << "Usage: " << argv[0] << " <label> <snapshot> OR "
                  << "<resolution> <scenario> <snapshot> <key>" << std::endl;
        return std::make_tuple("", "", -1);
    }

    try {
        scenario = std::stoi(argv[2]);
        snapshot = std::stoi(argv[3]);
    } catch (const std::invalid_argument& e) {
        std::cerr << "Error: Scenario and snapshot must be integers." << std::endl;
        return std::make_tuple("", "", -1);
    }

    pathStream << "/scratch/jh2/hs9158/results/LMC_run_" << resolution << "_wind_scenario_" << scenario 
               << key << "/snapshot_" << std::setfill('0') << std::setw(3) << snapshot << ".hdf5";
    outpathStream << "/scratch/jh2/hs9158/results/LMC_run_" << resolution << "_wind_scenario_" << scenario << key;

    return std::make_tuple(pathStream.str(), outpathStream.str(), snapshot);
}


int main(int argc, char* argv[]) {
    auto [snapshotPath, outputpath, snapshotNumber] = CreateSnapshotPath(argc, argv);
    if (snapshotPath.empty()) {
        return 1; // Exit if there was an error
    }

    std::cout << "Snapshot path: " << snapshotPath << std::endl;
    std::cout << "Snapshot number: " << snapshotNumber << std::endl;
    std::string groupname = "PartType4";

    auto hdf5Data = readHDF5(snapshotPath, groupname);

    double current_simulation_time = 0.0;
    try {
        if (argc == 3) {
            // Two arguments: label (e.g., MW) and snapshot (e.g., 23)
            int snapshot = std::stoi(argv[2]);
            current_simulation_time = 100 + snapshot * 1;
        }
        else if (argc == 4) {
            current_simulation_time = std::stod(argv[argc - 1]) * 5;
        }
        else if (argc == 5) {
            current_simulation_time = std::stod(argv[argc - 2]) * 5;
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Error converting argument to number: " << e.what() << std::endl;
    }

    std::cout << "Current simulation time: " << current_simulation_time << " Myr" << std::endl;

    // Create SlugWrapper objects and get ionizing luminosities for all particles
    std::vector<double> ionizingLuminosities = createSlugWrappersAndGetQH0(hdf5Data, current_simulation_time);

    std::cout << "Total size of ionizing luminosities vector: " << ionizingLuminosities.size() << std::endl;
    std::cout << "Median of ionizing luminosities vector: " << calculateMedian(ionizingLuminosities) << std::endl;

    // print the snapshot number
    std::cout << "Snapshot number: " << snapshotNumber << std::endl;

    std::ostringstream oss;
    oss << std::setfill('0') << std::setw(3) << snapshotNumber;
    std::string outputFileName = outputpath + "/ionizing_luminosities_" + oss.str() + ".txt";

    // Open output file
    std::ofstream outfile;

    outfile.open(outputFileName, std::ios_base::trunc); // overwrite
    if (!outfile.is_open()) {
        std::cerr << "Failed to open output file: " << outputFileName << std::endl;
        return 1;
    }
    // Extract ParticleIDs from hdf5 data and make sure they are saved as 16 bit integers without loss of precision
    auto it_int = std::find_if(hdf5Data.begin(), hdf5Data.end(),
                               [](const auto& pair) { return pair.first == "ParticleIDs"; });

    if (it_int != hdf5Data.end()) {
        const std::vector<double>& particleIDs = it_int->second;
        size_t numParticles = particleIDs.size();
        for (size_t i = 0; i < numParticles; ++i) {
            outfile << static_cast<int>(particleIDs[i]) << " " << ionizingLuminosities[i] << std::endl;
        }
        // std::cout << "First 10 elements of ParticleIDs and ionizingLuminosities:" << std::endl;
        // for (size_t i = 0; i < 10; ++i) {
        //     std::cout << static_cast<int>(particleIDs[i]) << " " << ionizingLuminosities[i] << std::endl;
    
    } else {
        std::cerr << "Error: ParticleIDs dataset not found." << std::endl;
    }
    std::cout << "Ionizing luminosities saved to: " << outputFileName << std::endl;

    // print the first 10 elements of ParticleIDs and ionizingLuminosities


    return 0;
}
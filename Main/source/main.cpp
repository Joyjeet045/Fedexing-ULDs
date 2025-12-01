#include <bits/stdc++.h>
#include "packing_structures.h"
#include "bin_pack.h"

using namespace std;

const int INF = INT_MAX;
const double MIN_BASE_CONTACT = 0.00;

int N, M, P; // Number of ULDs, packages, penalty multiplier
vector<ULD> ulds;
vector<Package> packages;

template <typename T>
class MaxQueue {
public:
  inline T max() const {
    return q.front().first;
  }
  inline T count() const {
    return q.front().second;
  }

  void insert(T val, int cnt = 1) {
    while (!q.empty() && q.back().first < val) {
      q.pop_back();
    }
    if(!q.empty() && q.back().first == val) q.back().second += cnt;
    else q.push_back({val, cnt});
  }

  void remove(T val, int cnt = 1) {
    if (!q.empty() && q.front().first == val) {
      q.front().second -= cnt;
      if(q.front().second == 0) q.pop_front();
    }
  }

private:
  std::deque<pair<T, int>> q;
};

template <typename T>
auto get_max_in_window(const vector<vector<T>>& a, std::size_t n1, std::size_t m1) {
  std::size_t n = a.size(), m = a[0].size();

  vector<vector<pair<T, int>>> res(n, vector<pair<T, int>>(m));
  for (std::size_t x = 0; x < n ; ++x) {
    MaxQueue<T> q;
    for (std::size_t y = 0; y < m1; ++y) {
      q.insert(a[x][y]);
    }

    for (std::size_t j = 0; j <= m - m1; ++j) {
      res[x][j] = {q.max(), q.count()};
      q.remove(a[x][j]);
      if (j + m1 < m) q.insert(a[x][j + m1]);
    }
  }

  for (std::size_t y = 0; y <= m - m1; ++y) {
    MaxQueue<T> q;
    for (std::size_t x = 0; x < n1; ++x) {
      q.insert(res[x][y].first, res[x][y].second);
    }

    for (std::size_t x = 0; x <= n - n1; ++x) {
      auto val = res[x][y];
      res[x][y] = {q.max(), q.count()};
      q.remove(val.first, val.second);
      if (x + n1 < n) q.insert(res[x + n1][y].first, res[x + n1][y].second);
    }
  }

  return res;
}


bool checkIntersection(const PkgPosition& pkg1, const PkgPosition& pkg2) {
    bool x_intersect = pkg1.x1 > pkg2.x0 && pkg2.x1 > pkg1.x0;
    bool y_intersect = pkg1.y1 > pkg2.y0 && pkg2.y1 > pkg1.y0;
    bool z_intersect = pkg1.z1 > pkg2.z0 && pkg2.z1 > pkg1.z0;
    return x_intersect && y_intersect && z_intersect;
}



struct PackingDetails {
    int totalCost = INF;
    int totalPacked = 0;
    int priorityULDs = 0;
    vector<Package> packages;
    vector<ULD> ulds;
    string method = "random";

    void printDetails(string outputFileName) {
        // Write output to file
        ofstream outputFile(outputFileName);
        if (!outputFile.is_open()) {
            cerr << "Error: Could not open " << outputFileName << " for writing.\n";
            return;
        }

        sort(packages.begin(), packages.end(), [](const Package& a, const Package& b) {
            return a.id < b.id;
        });

        outputFile << totalCost << "," << totalPacked << "," << priorityULDs << "\n";

        for (const auto& pkg : packages) {
            if (pkg.isPacked) {
                outputFile << "P-" << pkg.id << "," << "U" << pkg.uldID << ","
                        << pkg.position.x0 << "," << pkg.position.y0 << "," << pkg.position.z0 << ","
                        << pkg.position.x1 << "," << pkg.position.y1 << "," << pkg.position.z1 << "\n";
            } else {
                outputFile << "P-" << pkg.id << ",NONE,-1,-1,-1,-1,-1,-1\n";
            }
        }

        outputFile.close();
    }
    void printExtraDetails(string outputFileName) {
        // Write output to file
        ofstream outputFile(outputFileName);
        if (!outputFile.is_open()) {
            cerr << "Error: Could not open " << outputFileName << " for writing.\n";
            return;
        }

        outputFile << totalCost << "," << totalPacked << "," << priorityULDs << "\n";

        for (const auto& pkg : packages) {
            if (pkg.isPacked) {
                outputFile << pkg.id << "," << pkg.uldID << ","
                        << pkg.position.x0 << "," << pkg.position.y0 << "," << pkg.position.z0 << ","
                        << pkg.position.x1 << "," << pkg.position.y1 << "," << pkg.position.z1 << "\n";
            } else {
                outputFile << pkg.id << ",NONE,-1,-1,-1,-1,-1,-1\n";
            }
        }

        outputFile << "\nSorting method used : " << method << '\n';
        outputFile << "minimum base contact ratio used : " << MIN_BASE_CONTACT << '\n';
        outputFile << "\nPacking efficiency (by Volume) :-\n";
        double totVol = 0, usedvol = 0;
        for(auto& uld: ulds) {
            outputFile << uld.id << " : " << (100 - uld.volumeLeft / uld.volume * 100) << "%\n";
            totVol += uld.volume;
            usedvol += uld.volume - uld.volumeLeft;
        }
        outputFile << "net : " << usedvol / totVol * 100 << "%\n";

        outputFile << "\nPacking efficiency (by weight) :-\n";
        double totWt = 0, usedWt = 0;
        for(auto& uld: ulds) {
            outputFile << uld.id << " : " << uld.currentWeight / uld.maxWeight * 100 << "%\n";
            totWt += uld.maxWeight;
            usedWt += uld.currentWeight;
        }
        outputFile << "net : " << usedWt / totWt * 100 << "%\n";

        outputFile.close();
    }
} bestPacking;

void readInputFile() {
    string input;
    getline(cin, input); // Read the entire line of input
    stringstream ss(input);
    string token;
    getline(ss, token, ',');
    N = stoi(token);
    getline(ss, token, ',');
    M = stoi(token);
    getline(ss, token, ',');
    P = stoi(token);

    
    for (int i = 0; i < N; ++i) {
        string input;
        getline(cin, input); // Read the entire line of input
        stringstream ss(input);
        string token;

        getline(ss, token, ',');
        token = token.substr(1);
        int id = stoi(token);
        getline(ss, token, ',');
        double length = stod(token);
        getline(ss, token, ',');
        double width = stod(token);
        getline(ss, token, ',');
        double height = stod(token);
        getline(ss, token, ',');
        double maxWeight = stod(token);

        ulds.emplace_back(id, length, width, height, maxWeight);
    }

    for (int i = 0; i < M; ++i) {
        string input;
        getline(cin, input); // Read the entire line of input
        stringstream ss(input);
        string token;

        getline(ss, token, ',');
        token = token.substr(2);
        int id = stoi(token);
        getline(ss, token, ',');
        double length = stod(token);
        getline(ss, token, ',');
        double width = stod(token);
        getline(ss, token, ',');
        double height = stod(token);
        getline(ss, token, ',');
        double weight = stod(token);
        getline(ss, token, ',');
        char type = token[0];
        getline(ss, token, ',');
        double extraCost = -1;
        if(type=='E') extraCost = stod(token);

        packages.emplace_back(id, length, width, height, weight, type, extraCost);
    }
}

void computeDetails(vector<Package> &packages, vector<ULD> &ulds, int P) {
    PackingDetails currPacking;
    currPacking.totalCost = 0;

    for (auto& pkg : packages) {
        if (pkg.isPacked) {
            currPacking.totalPacked++;
            pkg.length = pkg.position.x1 - pkg.position.x0;
            pkg.width = pkg.position.y1 - pkg.position.y0;
            pkg.height = pkg.position.z1 - pkg.position.z0;
            for(auto& uld : ulds) {
                if(uld.id == pkg.uldID) {
                    uld.occupySpace(pkg, pkg.position.x0, pkg.position.y0, pkg.position.z0);
                    break;
                }
            }
        }
        else {
            if(pkg.type == 'P') {
                cerr << "Priority not packed\n";
                return;
            }
            else currPacking.totalCost += pkg.extraCost; // Add cost for Economy packages
        }
    }

    for (const auto& uld : ulds) {
        if (uld.hasPriorityPackage) {
            currPacking.priorityULDs++;
        }
    }

    // Add penalty cost for spreading Priority Packages across multiple ULDs
    currPacking.totalCost += P * currPacking.priorityULDs;

    if(bestPacking.totalCost > currPacking.totalCost) {
        currPacking.packages = packages;
        currPacking.ulds = ulds;
        bestPacking = currPacking;
    }
}


// Function to sort packages by different methods
void sortPackages(vector<Package>& packages, const string& method) {
    if (method == "extraCost") {
        // Priority first, then smallest extra cost
        sort(packages.begin(), packages.end(), [](const Package& a, const Package& b) {
            if (a.type != b.type) return a.type == 'P'; // Priority first

            return a.extraCost > b.extraCost; // largest extra cost
        });
    } else if (method == "weight") {
        // Priority first, then smallest weight
        sort(packages.begin(), packages.end(), [](const Package& a, const Package& b) {
            if (a.type != b.type) return a.type == 'P'; // Priority first
            return a.weight > b.weight; // Largest weight
        });
    } else if (method == "volume") {
        // Priority first, then largest volume
        sort(packages.begin(), packages.end(), [](const Package& a, const Package& b) {
            if (a.type != b.type) return a.type == 'P'; // Priority first
            return a.volume > b.volume; // Largest volume
        });
    } else if (method == "height") {
        // Priority first, then largest volume
        sort(packages.begin(), packages.end(), [](const Package& a, const Package& b) {
            if (a.type != b.type) return a.type == 'P'; // Priority first
            return a.height > b.height; // Largest height
        });
    } else if (method == "area") {
        // Priority first, then largest volume
        sort(packages.begin(), packages.end(), [](const Package& a, const Package& b) {
            if (a.type != b.type) return a.type == 'P'; // Priority first
            return a.length * a.width > b.length * b.width; // Largest surface area
        });
    } else if (method == "costPerVolume") {
        // Priority first, then largest cost per unit volume
        sort(packages.begin(), packages.end(), [](const Package& a, const Package& b) {
            if (a.type != b.type) return a.type == 'P'; // Priority first
            return (double) a.extraCost / a.volume > (double) b.extraCost / b.volume; // largest extra cost per unit volume
        });
    } else if (method == "costPerWeight") {
        // Priority first, then largest cost per unit weight
        sort(packages.begin(), packages.end(), [](const Package& a, const Package& b) {
            if (a.type != b.type) return a.type == 'P'; // Priority first
            return (double) a.extraCost / a.weight > (double) b.extraCost / b.weight; // largest extra cost per unit weight
        });
    }
    
}
void applyGravity(vector<Package>& current_packages) {
    // This ensures we settle the bottom-most items before the top ones
    sort(current_packages.begin(), current_packages.end(), [](const Package& a, const Package& b) {
        return a.position.z0 < b.position.z0;
    });

    for (auto& pkg : current_packages) {
        if (!pkg.isPacked) continue;

        while (pkg.position.z0 > 0) {
            PkgPosition nextPos = pkg.position;
            nextPos.z0--;
            nextPos.z1--;

            bool collision = false;
            for (const auto& other : current_packages) {
                if (pkg.id == other.id || !other.isPacked || pkg.uldID != other.uldID) continue;

                if (checkIntersection(nextPos, other.position)) {
                    collision = true;
                    break;
                }
            }

            if (!collision) {
                pkg.position = nextPos;
            } else {
                break; 
            }
        }
    }
}
// Function to check if a package fits into a ULD (considering weight and space)
bool fits(ULD& uld, Package& pkg) {

    auto z_max = get_max_in_window(uld.heightMap, pkg.length, pkg.width);
    

    // Check all possible positions for enough space
    double min_contact = MIN_BASE_CONTACT * pkg.length * pkg.width;
    for (int x = 0; x <= uld.length - pkg.length; x++) {
        for (int y = 0; y <= uld.width - pkg.width; y++) {
            if (z_max[x][y].first + pkg.height <= uld.height && z_max[x][y].second > min_contact) {
                uld.occupySpace(pkg, x, y, z_max[x][y].first); // Mark space as occupied
                return true;
            }
        }
    }
    

    return false; // No suitable space found
}

// First fit function to fit extra packages in remaining space
void firstFit(vector<Package> packages, vector<ULD> ulds, int P) {
    sort(ulds.begin(), ulds.end(), [](ULD& a, ULD& b) {
        return a.volume > b.volume;
    });

    // First-Fit Decreasing algorithm
    for (auto& pkg : packages) {
        if(pkg.isPacked) continue;

        vector<int> originalDims = {pkg.length, pkg.width, pkg.height};
        for (auto& uld : ulds) {
            if(uld.volumeLeft < pkg.volume) continue;
            if(uld.currentWeight + pkg.weight > uld.maxWeight) continue;

            auto dims = originalDims;
            do {
                pkg.length = dims[0];
                pkg.width = dims[1];
                pkg.height = dims[2];
                if (fits(uld, pkg)) {
                    break; // Package packed
                }
                next_permutation(dims.begin(), dims.end());
            } while(dims != originalDims);

            if(pkg.isPacked) break;
        }
        
    }

    // Calculate statistics
    computeDetails(packages, ulds, P); 
}

void readOutput(string output_file, vector<Package> &packages, vector<ULD> &ulds) {
    ifstream output(output_file);

    if (!output) {
        cerr << "Error output file.\n";
        return;
    }

    string first_line_output;
    getline(output, first_line_output);

    int m = packages.size();
    for (int i = 0; i < m; ++i) {
        string line;
        getline(output, line);
        stringstream ss(line);

        int pkg_id, uld_id;
        ss >> pkg_id;
        ss.ignore(1);
        ss >> uld_id;
        ss.ignore(1);

        int x_min, y_min, z_min, x_max, y_max, z_max;
        ss >> x_min;
        ss.ignore(1); // Skip comma
        ss >> y_min;
        ss.ignore(1); // Skip comma
        ss >> z_min;
        ss.ignore(1); // Skip comma
        ss >> x_max;
        ss.ignore(1); // Skip comma
        ss >> y_max;
        ss.ignore(1); // Skip comma
        ss >> z_max;

        if (uld_id == 0) continue;
        auto &pkg = packages[pkg_id-1];
        pkg.length = x_max - x_min;
        pkg.width = y_max - y_min;
        pkg.height = z_max - z_min;
        // pkg.position = PkgPosition(x_min, y_min, z_min, x_max, y_max, z_max);
        // pkg.uldID = uld_id;
        // pkg.isPacked = 1;
        auto &uld = ulds[uld_id-1];
        // cout<<pkg_id<<' '<<uld_id<<'\n';
        uld.occupySpace(pkg, x_min, y_min, z_min); // Mark space as occupied

    }
}

bool canMakeStable(int deltaX, int deltaY, Package& pkg, ULD& uld, vector<Package>& packages) {
     // Save original position
    auto originalPosition = pkg.position;

    // Move the package by deltaX and deltaY
    pkg.position.x0 += deltaX;
    pkg.position.x1 += deltaX;
    pkg.position.y0 += deltaY;
    pkg.position.y1 += deltaY;

    // Check if the package is out of ULD bounds
    if (pkg.position.x0 < 0 || pkg.position.x1 > uld.length ||
        pkg.position.y0 < 0 || pkg.position.y1 > uld.width) {
        // Revert to original position if out of bounds
        pkg.position = originalPosition;
        return false;
    }

    bool intersects = false;
    for (const auto& otherPkg : packages) {
        if (pkg.id == otherPkg.id || pkg.uldID != otherPkg.uldID) continue;

        if (checkIntersection(pkg.position, otherPkg.position)) {
            pkg.position = originalPosition;
            intersects = true;
            break;
        }
    }
    
    if(intersects) return false;

    // Try moving down until stability
    while (true) {
        int newZ = pkg.position.z0 - 1;
        if (newZ < 0) {
            // pkg.position.z0 = 0;
            // pkg.position.z1 = pkg.height;
            break;
        }

        // Tentative new position
        PkgPosition tentativePosition(
            pkg.position.x0, pkg.position.x1,
            pkg.position.y0, pkg.position.y1,
            newZ, newZ + (pkg.position.z1 - pkg.position.z0)
        );

        // Check intersection with other packages
        bool canMoveDown = true;
        for (const auto& otherPkg : packages) {
            if (pkg.id == otherPkg.id || pkg.uldID != otherPkg.uldID) continue;

            if (checkIntersection(tentativePosition, otherPkg.position)) {
                canMoveDown = false;
                break;
            }
        }

        if (canMoveDown) {
            pkg.position.z1 = newZ + (pkg.position.z1 - pkg.position.z0);
            pkg.position.z0 = newZ;
        } else {
            break;
        }
    }

    // Check centroid support
    double centroidX = (pkg.position.x0 + pkg.position.x1) / 2.0;
    double centroidY = (pkg.position.y0 + pkg.position.y1) / 2.0;
    bool centroidSupported = false;

    // Check if centroid is supported by either the ULD base or other packages
    if (pkg.position.z0 == 0) {
        centroidSupported = true; // Directly on the ULD base
    } else {
        for (const auto& otherPkg : packages) {
            if (pkg.id == otherPkg.id || pkg.uldID != otherPkg.uldID) continue;

            if (otherPkg.position.z1 == pkg.position.z0) { // Check the top surface of otherPkg
                if (centroidX >= otherPkg.position.x0 && centroidX <= otherPkg.position.x1 &&
                    centroidY >= otherPkg.position.y0 && centroidY <= otherPkg.position.y1) {
                    centroidSupported = true;
                    break;
                }
            }
        }
    }

    // If centroid is supported, finalize position
    if (centroidSupported) {
        return true;
    } else {
        // Revert position if not stable
        pkg.position = originalPosition;
        return false;
    }
}

void makeStableWithDelta(PackingDetails& currPacking) {
    sort(currPacking.packages.begin(), currPacking.packages.end(), [&](Package p1, Package p2){
        return p1.position.z0 < p2.position.z0;
    });

    for (auto& pkg : currPacking.packages) {
        if(!pkg.isPacked) continue;
        bool isStable = false;

        // Find the corresponding ULD for this package
        auto it = find_if(currPacking.ulds.begin(), currPacking.ulds.end(),
                               [&pkg](const ULD& uld) { return uld.id == pkg.uldID; });
        if (it == currPacking.ulds.end()) {
            cout << "Error: ULD with ID " << pkg.uldID << " not found for package " << pkg.id << endl;
            continue; // Skip this package if no matching ULD is found
        }
        auto& uld = *it; // Get the ULD object by reference

        // Iterate over all deltaX and deltaY values from -50 to 50
        for (int deltaX = 0; deltaX <= 50 && !isStable; ++deltaX) {
            for (int deltaY = 0; deltaY <= 50 && !isStable; ++deltaY) {
               if(canMakeStable(deltaX, deltaY, pkg, uld, currPacking.packages)) isStable = true;
            }
            for (int deltaY = -1; deltaY >= -50 && !isStable; --deltaY) {
               if(canMakeStable(deltaX, deltaY, pkg, uld, currPacking.packages)) isStable = true;
            }
        }
        for (int deltaX = -1; deltaX >= -50 && !isStable; --deltaX) {
            for (int deltaY = 0; deltaY <= 50 && !isStable; ++deltaY) {
               if(canMakeStable(deltaX, deltaY, pkg, uld, currPacking.packages)) isStable = true;
            }
            for (int deltaY = -1; deltaY >= -50 && !isStable; --deltaY) {
               if(canMakeStable(deltaX, deltaY, pkg, uld, currPacking.packages)) isStable = true;
            }
        }
    }
}


void packMultiple(vector<Package>& packages, vector<ULD>& ulds) {
    sort(ulds.begin(), ulds.end(), [](ULD& a, ULD& b) {
        return a.volume < b.volume;
    });
    vector<int> perm;
    for(int i = 0; i < (int) ulds.size(); i++) {
        perm.push_back(i);
    }
    int bestCost = -1;
    vector<Package> bestPack;
    do {
        vector<Package> packed, remainingPkgs = packages;
        for(auto& i : perm) {
            auto& uld = ulds[i];

            Binpack ob(remainingPkgs, uld);
            vector<Package> temp; 
            for(auto pkg: remainingPkgs) {
                if(!pkg.isPacked) {
                    temp.push_back(pkg);
                }
                else {
                    packed.push_back(pkg);
                }
            }
            remainingPkgs = temp;
        }

        int cost = 0;
        for(auto pkg : packed) {
            if(pkg.type == 'P') cost += inf;
            else cost += pkg.extraCost;
        }
        if(bestCost == -1 || bestCost < cost) {
            bestCost = cost;
            bestPack = packed;
            bestPack.insert(bestPack.end(), remainingPkgs.begin(), remainingPkgs.end());
        }
        if(remainingPkgs.empty()) break;
    }
    while (next_permutation(perm.begin(), perm.end()));

    packages = bestPack;
}

void packMask(int mask, vector<Package> packages, vector<ULD> ulds, int P) {
    int totPriorityUldVol = 0, totPriorityPkgVol = 0;
    vector<ULD> pulds, eulds;
    for(auto& uld : ulds) {
        int i = uld.id - 1;
        if(mask & (1 << i)) {
            pulds.push_back(uld);
            totPriorityUldVol += uld.volume;
        }
        else {
            eulds.push_back(uld);
        }
    }

    vector<Package> ppkgs, epkgs;
    for(auto& pkg: packages) {
        if(pkg.type == 'P') {
            ppkgs.push_back(pkg);
            totPriorityPkgVol += pkg.volume;
        }
        else {
            epkgs.push_back(pkg);
        }
    }

    if(totPriorityPkgVol > totPriorityUldVol) return;

    packMultiple(ppkgs, pulds);
    for(auto& pkg : ppkgs) {
        if(!pkg.isPacked) return;
    }

    auto rem_epkgs = epkgs;
    vector<Package> packed = ppkgs;

    for(auto uld : eulds) {
        //pack economy
        vector<Package> pkgs;
        int volSum = 0;
        for(auto pkg: rem_epkgs) {
            volSum += pkg.volume;
            if(volSum > 1.25 * uld.volume) break;
            pkgs.push_back(pkg);
        }
        Binpack ob(pkgs, uld);

        vector<Package> temp; 
        for(auto pkg: pkgs) {
            if(!pkg.isPacked) {
                temp.push_back(pkg);
            }
            else {
                packed.push_back(pkg);
            }
        }
        for(int i = pkgs.size(); i < (int) rem_epkgs.size(); i++) {
            temp.push_back(rem_epkgs[i]);
        }
        rem_epkgs = temp;
    }

    packed.insert(packed.end(), rem_epkgs.begin(), rem_epkgs.end());
    applyGravity(packed);
    computeDetails(packed, ulds, P);
}

int main() {
    freopen("input.txt", "r" , stdin);
    readInputFile();

    sortPackages(packages, "costPerVolume");

    for(int mask = 0; mask < (1 << N); mask++) {
        packMask(mask, packages, ulds, P);
    }

    makeStableWithDelta(bestPacking);
    packages = bestPacking.packages;
    ulds = bestPacking.ulds;
    sortPackages(packages, "costPerVolume");
    firstFit(packages, ulds, P);
    makeStableWithDelta(bestPacking);
    bestPacking.printDetails("output.txt");

    return 0;
}


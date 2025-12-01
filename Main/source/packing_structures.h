#include <bits/stdc++.h>

using namespace std;

#pragma once

struct PkgPosition {
    int x0, x1;
    int y0, y1;
    int z0, z1;
    PkgPosition(double x0, double x1, double y0, double y1, double z0,
                double z1)
        : x0(x0), x1(x1), y0(y0), y1(y1), z0(z0), z1(z1) {}
};

// Structure to represent a package with dimensions, weight, and type
struct Package {
    int id;
    int length, width, height, weight, volume, packx, packy, packz;
    char type; // 'P' for Priority, 'E' for Economy
    double extraCost;
    bool isPacked = false; // To check if the package is packed
    int uldID = -1;        // ULD ID where the package is placed
    PkgPosition position = PkgPosition(-1, -1, -1, -1, -1, -1); // Coordinates

    Package(int id = -1, int l = 0, int w = 0, int h = 0, int wt = 0,
            char t = 'E', double cost = 0)
        : id(id), length(l), width(w), height(h), weight(wt), volume(l * w * h),
          type(t), extraCost(cost) {}
};

// Structure to represent a ULD with dimensions, weight limit, and current load
struct ULD {
    int id;
    double length, width, height, maxWeight, volume;
    double currentWeight, volumeLeft;
    bool hasPriorityPackage = false; // To check if ULD contains Priority Package
    vector<vector<double>> heightMap; // 2D height map vector
    vector<Package> packages;

    ULD(int id = -1, double l = 0, double w = 0, double h = 0, double maxWt = 0)
        : id(id), length(l), width(w), height(h), maxWeight(maxWt),
          volume(l * w * h), currentWeight(0), volumeLeft(l * w * h) {
        // z_min.assign(length, vector<int> (width, 0));
        heightMap =
            vector<vector<double>>(static_cast<int>(length),
                                   vector<double>(static_cast<int>(width), 0));
    }

    // Mark space as occupied
    void occupySpace(Package &pkg, int x, int y, int z) {
        currentWeight += pkg.weight;
        for (int i = x; i < x + pkg.length && i < length; ++i) {
            for (int j = y; j < y + pkg.width && j < width; ++j) {
                heightMap[i][j] = max(heightMap[i][j], (double)z + pkg.height);
            }
        }

        // Update package's placement details
        pkg.isPacked = true;
        pkg.uldID = id;
        pkg.position.x0 = x;
        pkg.position.y0 = y;
        pkg.position.z0 = z;
        pkg.position.x1 = x + pkg.length;
        pkg.position.y1 = y + pkg.width;
        pkg.position.z1 = z + pkg.height;

        volumeLeft -= pkg.volume;

        if (pkg.type == 'P') {
            hasPriorityPackage = true;
        }
    }
};
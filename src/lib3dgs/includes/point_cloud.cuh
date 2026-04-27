#pragma once

#include <vector>

struct Point {
    float x;
    float y;
    float z;
    // Point& operator=(const Point& other) {
    //     if (this != &other) {
    //         x = other.x;
    //         y = other.y;
    //         z = other.z;
    //     }
    //     return *this;
    // }
};

struct Normal {
    float x;
    float y;
    float z;
    // Normal& operator=(const Normal& other) {
    //     if (this != &other) {
    //         x = other.x;
    //         y = other.y;
    //         z = other.z;
    //     }
    //     return *this;
    // }
};

// struct Color {
//     unsigned char r;
//     unsigned char g;
//     unsigned char b;
// };
struct Color {
    float r;
    float g;
    float b;
    // Color& operator=(const Color& other) {
    //     if (this != &other) {
    //         r = other.r;
    //         g = other.g;
    //         b = other.b;
    //     }
    //     return *this;
    // }
};
struct Distance {
    float r1;
    float r2;
    float r3;
    // Distance& operator=(const Distance& other) {
    //     if (this != &other) {
    //         r1 = other.r1;
    //         r2 = other.r2;
    //         r3 = other.r3;
    //     }
    //     return *this;
    // }
};

struct Quaternions {
    float qw;
    float qx;
    float qy;
    float qz;
    // Quaternions& operator=(const Quaternions& other) {
    //     if (this != &other) {
    //         qw = other.qw;
    //         qx = other.qx;
    //         qy = other.qy;
    //         qz = other.qz;
    //     }
    //     return *this;
    // }
};

struct PointCloud {
    std::vector<Point> _points;
    std::vector<Normal> _normals;
    std::vector<Distance> _distance;
    std::vector<Quaternions> _quaternion;
    std::vector<Color> _colors;
};

#pragma pack(push, 1)
struct GS_point {
    Point _points;
    Normal _normals;
    Distance _distance;
    Quaternions _quaternion;
    Color _colors;
    float _opacity;
    float index;
    float flag_in_fov;

    // GS_point& operator=(const GS_point& other) {
    //     if (this != &other) {
    //         _points = other._points;            
    //         _normals = other._normals;        
    //         _distance = other._distance;         
    //         _quaternion = other._quaternion;     
    //         _colors = other._colors;               
    //         index = other.index; 
    //     }  
    //     return *this;
    // }
};
#pragma pack(pop)
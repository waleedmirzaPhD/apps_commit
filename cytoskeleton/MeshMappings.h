#pragma once
#include <cmath>

inline void mapCircleToRegularPolygon(
    double x, double y, int N, double R, double rotation,
    double& X, double& Y)
{
    double r2 = x*x + y*y;
    double xr = x, yr = y;
    if (r2 > 1.0) {
        double r = std::sqrt(r2);
        if (r > 0.0) { xr = x / r; yr = y / r; }
    }
    double r = std::sqrt(xr*xr + yr*yr);
    if (r == 0.0) { X = 0.0; Y = 0.0; return; }

    const double ux = xr / r;
    const double uy = yr / r;
    double phi = std::atan2(uy, ux) - rotation;

    const double twoPiOverN = 2.0 * M_PI / double(N);
    const double piOverN    = M_PI / double(N);
    double alpha = std::fmod(phi + piOverN, twoPiOverN);
    if (alpha < 0.0) alpha += twoPiOverN;
    alpha -= piOverN;

    const double rho = R * std::cos(piOverN) / std::cos(alpha);
    const double s   = r * rho;
    const double cR  = std::cos(rotation);
    const double sR  = std::sin(rotation);
    X = s * (ux * cR - uy * sR);
    Y = s * (ux * sR + uy * cR);
}

inline void mapCircleToTriangle(double x, double y, double R,
                                double& X, double& Y,
                                double rotation = M_PI / 2.0)
{
    mapCircleToRegularPolygon(x, y, 3, R, rotation, X, Y);
}

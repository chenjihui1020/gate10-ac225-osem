//图像重建
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <chrono>

#define pi 3.14159265359
#define scarad 43.7//42。5+2
#define absrad 68.7//71+4.5
#define rotdir -1.0
using namespace std;
__device__ double atomicAdd1(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
            __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
__device__ void position(int d, int x1, int z1, int scaxy, int absdoi, double pos[3], int ii, int jj)
{
    double x;
    double y;
    double z;

    double xx;
    double yy;

    x = 11.2 - (double)x1 * 3.2;
    y = scarad;// +0.75;//40.75;// + 6.5;
    z = 11.2 - (double)z1 * 3.2;

    if (d < 8)
    {
        int xx1 = ii / scaxy;
        int zz1 = ii - scaxy * xx1;
        x = 11.2 - (double)x1 * 3.2 - 1.25 + 2.5 / scaxy / 2 + 2.5 / scaxy * xx1;
        z = 11.2 - (double)z1 * 3.2 - 1.25 + 2.5 / scaxy / 2 + 2.5 / scaxy * zz1;
    }

    if (d >= 8)
        y = absrad +(double)9. / absdoi / 2 + (double)9. / absdoi * jj;//62.5 + (double)9./absdoi / 2 + (double)9./absdoi * jj;// + 6.5;
    
    xx = x * cos(1.0 * (double)d * rotdir * pi / 4) + y * sin(1.0 * (double)d * rotdir * pi / 4);
    yy = -x * sin(1.0 * (double)d * rotdir * pi / 4) + y * cos(1.0 * (double)d * rotdir * pi / 4);

    pos[0] = xx;
    pos[1] = yy;
    pos[2] = z;
}

__device__ double cos_cal(double vec1[3], double vec2[3])
{
    double abs1 = sqrt(pow(vec1[0], 2) + pow(vec1[1], 2) + pow(vec1[2], 2));
    double abs2 = sqrt(pow(vec2[0], 2) + pow(vec2[1], 2) + pow(vec2[2], 2));
    double r = (vec1[0] * vec2[0] + vec1[1] * vec2[1] + vec1[2] * vec2[2]) / abs1 / abs2;
    return r;
}

__device__ double sig_th(double E_inc, double E_abs, double E_sca, double th)
{
    /*
    double sig1 = E_abs * 0.1 / 2.355;
    double sig = sig1*0.511/pow(E_abs,2)/sin(th);
    */

    double sig1 = E_sca * 0.1 / 2.355 * sqrt(511 / E_sca);
    double sig2 = E_abs * 0.1 / 2.355 * sqrt(511 / E_abs);
    double sig = (double)511. / pow(E_inc, 2) / sin(th) * sqrt(pow(sig1, 2) + pow(E_sca, 2) * pow(E_abs + E_inc, 2) / pow(E_abs, 4) * pow(sig2, 2));

    return sig;
}
__device__ double compton(double E_inc, double costh)
{
    double E_sca = E_inc / (1 + E_inc / 511 * (1 - costh));
    return E_sca;
}
__device__ double Klein_Nishina(double E_inc, double E_sca, double costh)
{
    double factor;
    factor = 0.1386;
    
    double P = factor * pow(E_sca / E_inc, 2) * (E_sca / E_inc + E_inc / E_sca - (1 - pow(costh, 2)));
    return P;
}

__device__ void cross_product(double* vec1, double* vec2, double* vec3)
{
    vec3[0] = vec1[1] * vec2[2] - vec1[2] * vec2[1];
    vec3[1] = -(vec1[0] * vec2[2] - vec1[2] * vec2[0]);
    vec3[2] = vec1[0] * vec2[1] - vec1[1] * vec2[0];

    double norm = sqrt(pow(vec3[0],2)+ pow(vec3[1], 2) + pow(vec3[2], 2));

    for (int i = 0; i < 3; i++)
    {
        vec3[i] = vec3[i] / norm;
    }
}

__device__ double th_weight(double sig, double th_c, double th)
{
    double w;
    w = 1 / sig / sqrt(2 * pi) * exp(-pow((th_c - th) / sig, 2) / 2);

    return w;
}


__global__ void forward_proj(double* sum1, double* image, double* CS, int* d_inpu, double* Es, int inputsize, int x, int y, int z, double xsize, double ysize, double zsize, double E_inc, int scaxy, int absdoi, double scahighlim, double scalowlim)
{
    int q = blockIdx.x * blockDim.x + threadIdx.x;

    if (q < x * y * z)
    {
        int k = floor((double)q / x / y);
        int j = floor((double)q / x) - y * k;
        int i = q - x * j - x * y * k;

        double scavol = 2.5 * 2.5 * 4 / scaxy / scaxy;
        double absvol = 2.5 * 2.5 * 9. / absdoi;

        double sca_pos1[3], abs_pos1[3], sca_pos2[3], abs_pos2[3];
        double vec_c11[3], vec_c12[3], vec_c21[3], vec_c22[3];

        double yvec1[3], yvec2[3], yvec3[3], yvec4[3];

        double pox, poy, poz;
        double w1, w2, w_th1, w_th2, w_total;
        double costh_c1, costh_c2, th_c1, th_c2, th, sigma_th1, sigma_th2;
        double E_sca1, E_sca2, P, costh;
        double dist_vm1, dist_vv1, dist_vm2, dist_vv2, cosvm1, cosvv1, cosvm2, cosvv2;
        double sigma_det;
        double sig1, sig2;
        double dl1, dl2, dl3, dl4;
        double r1, r2, r3, r4;

        int x1, z1, x2, z2, x3, z3, x4, z4;
        

        pox = -xsize / 2 + xsize / x / 2 + xsize / x * i;
        poy = -ysize / 2 + ysize / y / 2 + ysize / y * j;
        poz = -zsize / 2 + zsize / z / 2 + zsize / z * k;

        for (int p = 0; p < inputsize; p++)
        {

            int d1 = d_inpu[p + inputsize * 0];
            int d2 = d_inpu[p + inputsize * 2];

            yvec1[0] = sin((double)1.0 * d1 * rotdir / 4 * pi);
            yvec1[1] = cos((double)1.0 * d1 * rotdir / 4 * pi);
            yvec1[2] = 0;

            yvec2[0] = sin((double)1.0 * d2 * rotdir / 4 * pi);
            yvec2[1] = cos((double)1.0 * d2 * rotdir / 4 * pi);
            yvec2[2] = 0;

            int ad1 = d_inpu[p + inputsize * 1];
            int ad2 = d_inpu[p + inputsize * 3];

            x1 = floor((double)ad1 / 8);
            z1 = ad1 - 8 * x1;

            x2 = floor((double)ad2 / 8);
            z2 = ad2 - 8 * x2;

            for (int ii = 0; ii < scaxy * scaxy; ii++)
            {
                for (int jj = 0; jj < absdoi; jj++)
                {
                    w_total = 0;
                    position(d1, x1, z1, scaxy, absdoi, sca_pos1, ii, jj);
                    position(d2, x2, z2, scaxy, absdoi, abs_pos1, ii, jj);


                    double val = 0;
                    double E1 = Es[p + inputsize * 0];
                    double E2 = Es[p + inputsize * 1];

                    double be1 = acos(1 - 511. * (1 / (E_inc - E1) - 1 / E_inc));

                    vec_c11[0] = (sca_pos1[0] - pox);
                    vec_c11[1] = (sca_pos1[1] - poy);
                    vec_c11[2] = (sca_pos1[2] - poz);

                    vec_c12[0] = (abs_pos1[0] - sca_pos1[0]);
                    vec_c12[1] = (abs_pos1[1] - sca_pos1[1]);
                    vec_c12[2] = (abs_pos1[2] - sca_pos1[2]);


                    cosvm1 = cos_cal(vec_c11, yvec1);
                    cosvv1 = cos_cal(vec_c12, yvec2);

                    cosvm1 = abs(cosvm1);
                    cosvv1 = abs(cosvv1);

                    costh_c1 = cos_cal(vec_c11, vec_c12);

                    dist_vm1 = (pow(vec_c11[0], 2) + pow(vec_c11[1], 2) + pow(vec_c11[2], 2));
                    dist_vv1 = (pow(vec_c12[0], 2) + pow(vec_c12[1], 2) + pow(vec_c12[2], 2));

                    if (costh_c1 >= 1.0)
                        costh_c1 = 0.99999;
                    if (costh_c1 <= -1.0)
                        costh_c1 = -0.99999;

                    th_c1 = acos(costh_c1);

                    sigma_det = 0;
                    E_sca1 = compton(E_inc, costh_c1);
                    /*
                    if (E_inc - E_sca1 > scahighlim)
                        continue;
                    if (E_inc - E_sca1 > scalowlim)
                        continue;
                     */
                    sigma_th1 = sig_th(E_inc, E_sca1, E_inc - E_sca1, be1);

                    sig1 = sigma_th1;

                    //if (abs(th_c1 - be1) > sig1 * 2.355)
                        //continue;

                    w_th1 = th_weight(sig1, th_c1, be1);

                    w_total = w_th1 / dist_vm1 / dist_vv1;

                    w_total = image[q] * w_total;


                    double KN1 = Klein_Nishina(E_inc, E_sca1, costh_c1);
                    KN1 = KN1 * CS[(int)floor(E_inc) - 1 + 1300 * 0] * CS[(int)floor(E_sca1) - 1 + 1300 * 1];

                    double lensca1 = 2 / cosvm1;
                    double lensca2 = 2 / cosvv1;
                    double lenabs = ((double)9. / absdoi / 2 + (double)9. / absdoi * jj) / cosvv1;

                    double w1 = exp(-lensca1 * CS[(int)floor(E_inc) - 1 + 1300 * 2]);
                    double w2 = exp(-lensca2 * CS[(int)floor(E_sca1) - 1 + 1300 * 2]);
                    double w3 = exp(-lenabs * CS[(int)floor(E_sca1) - 1 + 1300 * 2]);

                    w_total = w_total * KN1 * scavol * absvol * w1 * w2 * w3;

                    atomicAdd1(&sum1[p], w_total);
                }
            }



        }
    }

}


__global__ void backward_proj(double* sum2, double* sum1, double* CS, int* d_inpu, double* Es, int inputsize, int x, int y, int z, double xsize, double ysize, double zsize, double E_inc, int scaxy, int absdoi, double scahighlim, double scalowlim)
{
    int q = blockIdx.x * blockDim.x + threadIdx.x;

    if (q < x * y * z)
    {
        int k = floor((double)q / x / y);
        int j = floor((double)q / x) - y * k;
        int i = q - x * j - x * y * k;

        double scavol = 2.5 * 2.5 * 4 / scaxy / scaxy;
        double absvol = 2.5 * 2.5 * 9. / absdoi;

        double sca_pos1[3], abs_pos1[3], sca_pos2[3], abs_pos2[3];
        double vec_c11[3], vec_c12[3], vec_c21[3], vec_c22[3];

        double yvec1[3], yvec2[3], yvec3[3], yvec4[3];

        double pox, poy, poz;
        double w1, w2, w_th1, w_th2, w_total;
        double costh_c1, costh_c2, th_c1, th_c2, th, sigma_th1, sigma_th2;
        double E_sca1, E_sca2, P, costh;
        double dist_vm1, dist_vv1, dist_vm2, dist_vv2, cosvm1, cosvv1, cosvm2, cosvv2;
        double sigma_det;
        double sig1, sig2;
        double dl1, dl2, dl3, dl4;
        double r1, r2, r3, r4;

        int x1, z1, x2, z2, x3, z3, x4, z4;
        

        pox = -xsize / 2 + xsize / x / 2 + xsize / x * i;
        poy = -ysize / 2 + ysize / y / 2 + ysize / y * j;
        poz = -zsize / 2 + zsize / z / 2 + zsize / z * k;

        for (int p = 0; p < inputsize; p++)
        {
            if (sum1[p] != 0)
            {
                w_total = 0;
                int d1 = d_inpu[p + inputsize * 0];
                int d2 = d_inpu[p + inputsize * 2];

                yvec1[0] = sin((double)1.0 * d1 * rotdir / 4 * pi);
                yvec1[1] = cos((double)1.0 * d1 * rotdir / 4 * pi);
                yvec1[2] = 0;

                yvec2[0] = sin((double)1.0 * d2 * rotdir / 4 * pi);
                yvec2[1] = cos((double)1.0 * d2 * rotdir / 4 * pi);
                yvec2[2] = 0;


                int ad1 = d_inpu[p + inputsize * 1];
                int ad2 = d_inpu[p + inputsize * 3];

                x1 = floor((double)ad1 / 8);
                z1 = ad1 - 8 * x1;

                x2 = floor((double)ad2 / 8);
                z2 = ad2 - 8 * x2;

                for (int ii = 0; ii < scaxy * scaxy; ii++)
                {
                    for (int jj = 0; jj < absdoi; jj++)
                    {
                        position(d1, x1, z1, scaxy, absdoi, sca_pos1, ii, jj);
                        position(d2, x2, z2, scaxy, absdoi, abs_pos1, ii, jj);

                        double val = 0;
                        double E1 = Es[p + inputsize * 0];
                        double E2 = Es[p + inputsize * 1];

                        double be1 = acos(1 - 511. * (1 / (E_inc - E1) - 1 / E_inc));

                        vec_c11[0] = (sca_pos1[0] - pox);
                        vec_c11[1] = (sca_pos1[1] - poy);
                        vec_c11[2] = (sca_pos1[2] - poz);

                        vec_c12[0] = (abs_pos1[0] - sca_pos1[0]);
                        vec_c12[1] = (abs_pos1[1] - sca_pos1[1]);
                        vec_c12[2] = (abs_pos1[2] - sca_pos1[2]);


                        cosvm1 = cos_cal(vec_c11, yvec1);
                        cosvv1 = cos_cal(vec_c12, yvec2);

                        cosvm1 = abs(cosvm1);
                        cosvv1 = abs(cosvv1);

                        costh_c1 = cos_cal(vec_c11, vec_c12);


                        dist_vm1 = (pow(vec_c11[0], 2) + pow(vec_c11[1], 2) + pow(vec_c11[2], 2));
                        dist_vv1 = (pow(vec_c12[0], 2) + pow(vec_c12[1], 2) + pow(vec_c12[2], 2));


                        if (costh_c1 >= 1.0)
                            costh_c1 = 0.99999;
                        if (costh_c1 <= -1.0)
                            costh_c1 = -0.99999;

                        th_c1 = acos(costh_c1);

                        sigma_det = 0;
                        E_sca1 = compton(E_inc, costh_c1);
                        
                        /*
                        if (E_inc - E_sca1 > scahighlim)
                            continue;
                        if (E_inc - E_sca1 > scalowlim)
                            continue;
                        */
                        sigma_th1 = sig_th(E_inc, E_sca1, E_inc - E_sca1, be1);

                        sig1 = sigma_th1;

                        //if (abs(th_c1 - be1) > sig1 * 2.355)
                            //continue;

                        w_th1 = th_weight(sig1, th_c1, be1);

                        w_total = w_th1 / dist_vm1 / dist_vv1;


                        double KN1 = Klein_Nishina(E_inc, E_sca1, costh_c1);
                        KN1 = KN1 * CS[(int)floor(E_inc) - 1 + 1300 * 0] * CS[(int)floor(E_sca1) - 1 + 1300 * 1];

                        double lensca1 = 2 / cosvm1;
                        double lensca2 = 2 / cosvv1;
                        double lenabs = ((double)9. / absdoi / 2 + (double)9. / absdoi * jj) / cosvv1;

                        double w1 = exp(-lensca1 * CS[(int)floor(E_inc) - 1 + 1300 * 2]);
                        double w2 = exp(-lensca2 * CS[(int)floor(E_sca1) - 1 + 1300 * 2]);
                        double w3 = exp(-lenabs * CS[(int)floor(E_sca1) - 1 + 1300 * 2]);


                        w_total = w_total * KN1 * absvol * scavol * w1 * w2 * w3;

                        sum2[q] = sum2[q] + w_total / sum1[p];
                    }
                }
                
                

            }
        }
    }


    
}
__global__ void sens_maker(double* sensitivity, double* CS, int x, int y, int z, int subset, double xsize, double ysize, double zsize, double E_inc, int scaxy, int absdoi, double scahighlim, double scalowlim, int subsetnum)
{
    int q = blockIdx.x * blockDim.x + threadIdx.x;

    if (q < x * y * z* 64*64)
    {
        int p = floor((double)q / x / y / z);
        int k = floor((double)q / x / y) - z * p;
        int j = floor((double)q / x) - y * (k + z * p);
        int i = q - x * (j + y * (k + z * p));


        int d1 = subset;
        int ad1 = floor((double)p / 64);
        int d2 = d1 + 8;
        int ad2 = p - 64 * (ad1);


        double scavol = 2.5 * 2.5 * 4 / scaxy / scaxy;
        double absvol = 2.5 * 2.5 * 9. / absdoi;

        double pos1[3], pos2[3];
        double vec_c1[3], vec_c2[3];


        double pox, poy, poz;
        double w1, w2, w_th, w_total;
        double costh_c, th_c, th, sigma_th;
        double E_sca1, P, costh;
        double dist_vm1, dist_vm2,dist_vv1, cosvm1, cosvm2, cosvv1;
        double sigma_det;
        double sig;

        int x1, z1, x2, z2;

        pox = -xsize / 2 + xsize / x / 2 + xsize / x * i;
        poy = -ysize / 2 + ysize / y / 2 + ysize / y * j;
        poz = -zsize / 2 + zsize / z / 2 + zsize / z * k;
        double yvec1[3], yvec2[3];


        x1 = floor((double)ad1 / 8);
        z1 = ad1 - 8 * x1;

        x2 = floor((double)ad2 / 8);
        z2 = ad2 - 8 * x2;

        yvec1[0] = sin((double)1.0 * d1 * rotdir / 4 * pi);
        yvec1[1] = cos((double)1.0 * d1 * rotdir / 4 * pi);
        yvec1[2] = 0;

        for (int ii = 0; ii < scaxy * scaxy; ii++)
        {
            for (int jj = 0; jj < absdoi; jj++)
            {
                position(d1, x1, z1, scaxy, absdoi, pos1, ii, jj);
                position(d2, x2, z2, scaxy, absdoi, pos2, ii, jj);

                vec_c1[0] = (pos1[0] - pox);
                vec_c1[1] = (pos1[1] - poy);
                vec_c1[2] = (pos1[2] - poz);

                vec_c2[0] = (pos2[0] - pos1[0]);
                vec_c2[1] = (pos2[1] - pos1[1]);
                vec_c2[2] = (pos2[2] - pos1[2]);

                cosvm1 = cos_cal(vec_c1, yvec1);
                cosvm2 = cos_cal(vec_c2, yvec1);
                cosvv1 = cos_cal(vec_c2, vec_c1);

                cosvm1 = abs(cosvm1);
                cosvm2 = abs(cosvm2);
                cosvv1 = abs(cosvv1);

                E_sca1 = compton(E_inc, cosvv1);
                /*
                if (E_inc - E_sca1 > scahighlim+10.)
                    continue;
                if (E_inc - E_sca1 < scalowlim-10.)
                    continue;
                 */
                double lensca1 = 2 / cosvm1;
                double lensca2 = 2 / cosvv1;
                double lenabs = ((double)9. / absdoi / 2 + (double)9. / absdoi * jj) / cosvm2;

                double w1 = exp(-lensca1 * CS[(int)floor(E_inc) - 1 + 1300 * 2]);
                double w2 = exp(-lensca2 * CS[(int)floor(E_sca1) - 1 + 1300 * 2]);
                double w3 = exp(-lenabs * CS[(int)floor(E_sca1) - 1 + 1300 * 2]);

                double KN1 = Klein_Nishina(E_inc, E_sca1, cosvv1);
                KN1 = KN1 * CS[(int)floor(E_inc) - 1 + 1300 * 0] * CS[(int)floor(E_sca1) - 1 + 1300 * 1];

                dist_vm1 = (pow(vec_c1[0], 2) + pow(vec_c1[1], 2) + pow(vec_c1[2], 2));
                dist_vv1 = (pow(vec_c2[0], 2) + pow(vec_c2[1], 2) + pow(vec_c2[2], 2));

                w_total = KN1 / dist_vm1 / dist_vv1 * w1 * w2 * w3;
                w_total = w_total * scavol * absvol;


                //sensitivity[i+x*(j+y*k)] = sensitivity[i + x * (j + y * k)] + w_total;
                

                if (subsetnum == 1)
                    atomicAdd1(&sensitivity[i + x * (j + y * (k))], w_total);
                else
                    atomicAdd1(&sensitivity[i + x * (j + y * (k + z * subset))], w_total);
            }
        }
        
        
    }
    
}

__global__ void normal(double* image_o, double* image_t, double* sensitivity, double* sum2, int x, int y, int z, int subset)
{
    int q = blockIdx.x * blockDim.x + threadIdx.x;

    if (q < x * y * z && sensitivity[q + x * y * z * subset] != 0)
    {
        image_o[q] = image_t[q] / sensitivity[q + x * y * z * subset] * sum2[q];


        image_t[q] = image_o[q];
    }

    
}

int main()
{
    int senind = 1;
    int iternum = 30;
    int subsetnum = 8;
    double scalowlim = 5;//20
    double scahighlim = 140;

    double E_inc = 218;// 511;

    int device;
    cudaGetDevice(&device);
    struct cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);

    string data, line, inputfilename, totall;
    double* CS = new double[3 * 1300];
    
    ifstream CSin("CS_crystal.txt");
    for (int i = 0; i < 1300; i++)
    {
        getline(CSin, line, '\n');
        istringstream strm(line);
        getline(strm, data, ' ');
        getline(strm, data, ' ');
        CS[i + 1300 * 0] = stod(data) * 0.663;
        getline(strm, data, ' ');
        CS[i + 1300 * 1] = stod(data) * 0.663;
        getline(strm, data, ' ');
        CS[i + 1300 * 2] = stod(data) * 0.663;
        
    }


    int* dict = new int[512 * 1023];
    int c = 0;
    for (int i = 0; i < 1024; i++)
    {
        for (int j = i + 1; j < 1024; j++)
        {
            dict[c] = i + 1024 * j;
            c = c + 1;
        }
    }

    int x = 80;
    int y = 80;
    int z = 25;

    double xsize = 80;
    double ysize = 80;
    double zsize = 50;
    double* sensitivity = new double[x * y * z * subsetnum];

    int scaxy = 1;
    int absdoi =1;
    
    double mu = 0.06;
    //vector<double> image(x * y * z);
    double* image = new double[x * y * z];
    for (int i = 0; i < x; i++)
    {
        for (int j = 0; j < y; j++)
        {
            for (int k = 0; k < z; k++)
            {
                if (pow((double)i - (double)x / 2 + 0.5, 2) + pow((double)j - (double)y / 2 + 0.5, 2) < pow((double)x / 2 + 5,2))
                    image[i + x * (j + y * k)] = 1;
                else
                    image[i + x * (j + y * k)] = 0;
            }
        }
        
    }
    for (int i = 0; i < x * y * z * subsetnum; i++)
    {
        sensitivity[i] = 0;
    }

    int val;
    double val1;
    

    vector<vector<vector<int> > >tempinp(8);
    vector<vector<vector<double> > >tempEs(8);
    vector<int> inptemp(4);
    vector<double> Etemp(2);

    int l = 0;
    int ll = 0;
    unsigned int totallines = 0;
    ifstream inputfile("inputfilename.txt");
    getline(inputfile, inputfilename, '\n');
    cout << "Input File: " << inputfilename << endl;
    ifstream inp_path(inputfilename);

    string foldername;
    istringstream strm2(inputfilename);
    getline(strm2, foldername, '/');
    getline(strm2, foldername, '.');

    cout << "Output will be saved at " << foldername << endl;

    while (getline(inp_path, line, '\n'))
    {
        istringstream strm(line);

        for (int i = 0; i < 4; i++)
        {
            getline(strm, data, ' ');
            inptemp[i] = stoi(data);
        }
        
        for (int i = 0; i < 2; i++)
        {
            getline(strm, data, ' ');
            Etemp[i] = stod(data);
        }
        getline(strm, data, ' ');
        double dt = stod(data);
        

        double be1 = acos(1 - 511. * (1 / (E_inc - Etemp[0]) - 1 / E_inc));

        if (isnan(be1) == 1)
            continue;
        if (Etemp[0] > scahighlim)
            continue;
        if (Etemp[0] < scalowlim)
            continue;
        if (Etemp[0] + Etemp[1] < E_inc * 0.9 || Etemp[0] + Etemp[1] >E_inc * 1.1)
            continue;


        if (subsetnum == 1)
        {
            tempinp[0].push_back(inptemp);
            tempEs[0].push_back(Etemp);
        }
        else
        {
            tempinp[inptemp[0]].push_back(inptemp);
            tempEs[inptemp[0]].push_back(Etemp);
        }

       
        l++;

    }
    cout << l << " lines have been taken." << endl;

    
    double* sum2 = new double[x * y * z];
    double* image_temp = new double[x * y * z];

    

    for (int i = 0; i < x*y*z; i++)
    {
        sum2[i] = 0;
    }
    


    cudaError_t cudaStatus;

    

    //dim3 threadsPerBlock(5, 5, 2);
    //dim3 numblock(10, 10, 10);
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
    }
    size_t size = x * y * z * sizeof(double);

    double* sp = 0;

    double* d_image_t = 0;
    double* d_image_u = 0;

    double* sum2check = new double[x * y * z];
    int* d_dict = 0;
    double* d_CS = 0;
    double* d_KN_pol = 0;
    
    cudaStatus = cudaMalloc(&sp, x * y * z * subsetnum * sizeof(double));
    
    
    cudaStatus = cudaMalloc(&d_image_t, x * y * z * sizeof(double));
    cudaStatus = cudaMalloc(&d_image_u, x * y * z * sizeof(double));
    
    
    cudaStatus = cudaMalloc(&d_dict, 512 * 1023 * sizeof(int));
    cudaMalloc(&d_CS, 3 * 1300 * sizeof(double));


    
    cudaStatus = cudaMemcpy(sp, sensitivity, x*y*z *subsetnum* sizeof(double), cudaMemcpyHostToDevice);
    cudaStatus = cudaMemcpy(d_image_t, image, x* y* z * sizeof(double), cudaMemcpyHostToDevice);
    cudaStatus = cudaMemcpy(d_image_u, image, x* y* z * sizeof(double), cudaMemcpyHostToDevice);
    cudaStatus = cudaMemcpy(d_dict, dict, 512 * 1023 * sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(d_CS, CS, 3 * 1300 * sizeof(double), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int numblock = ceil(x * y * z* 64 * 64 / threadsPerBlock);

    if (senind == 1)
    {
        cout << "Making Sensitivity." << endl;
        for (int k = 0; k < 8/*subsetnum*/; k++)
        {
            sens_maker << <numblock, threadsPerBlock >> > (sp, d_CS, x, y, z, k, xsize, ysize, zsize, E_inc, scaxy, absdoi, scahighlim, scalowlim, subsetnum);
            cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            }
            cudaStatus = cudaDeviceSynchronize();
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
            }
            cout << k << endl;
        }

        cudaStatus = cudaMemcpy(sensitivity, sp, x * y * z * subsetnum * sizeof(double), cudaMemcpyDeviceToHost);

        ofstream sensout("sensitivity/sensitivity_subset.txt");

        for (int sub = 0; sub < subsetnum; sub++)
        {
            for (int k = 0; k < z; k++)
            {
                for (int j = 0; j < y; j++)
                {
                    for (int i = 0; i < x; i++)
                    {
                        sensout << sensitivity[i + x * (j + y * (k + z * sub))] << "\t";
                    }
                    sensout << endl;
                }
            }
        }
        
    }
    else
    {
        int senindex = 0;
        ifstream sensin("sensitivity/sensitivity_subset.txt");
        while (getline(sensin, line, '\n'))
        {
            istringstream strm(line);
            for (int i = 0; i < x; i++)
            {
                getline(strm, data, '\t');
                sensitivity[senindex] = stod(data);
                senindex++;
            }
        }
        cudaMemcpy(sp, sensitivity, x * y * z * subsetnum *sizeof(double), cudaMemcpyHostToDevice);
    }

    
    for (int iter = 0; iter < iternum; iter++)
    {
        
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        

        for (int subset = 0; subset < subsetnum; subset++)
        {
            l = tempinp[subset].size();
            int* addr = new int[l * 4];
            double* Es = new double[l * 2];
            double* sum1 = new double[l];
            double* d_sum1 = 0;
            double* d_sum2 = 0;
            int* d_inp = 0;
            double* d_Es = 0;

            for (int i = 0; i < l; i++)
            {
                for (int j = 0; j < 2; j++)
                {
                    Es[i + l * j] = tempEs[subset][i][j];
                }
                for (int k = 0; k < 4; k++)
                {
                    addr[i + l * k] = tempinp[subset][i][k];
                }
                sum1[i] = 0;
            }
            cudaStatus = cudaMalloc(&d_sum2, x * y * z * sizeof(double));
            cudaStatus = cudaMalloc(&d_sum1, l * sizeof(double));
            cudaStatus = cudaMalloc(&d_inp, l * 4 * sizeof(int));
            cudaStatus = cudaMalloc(&d_Es, l * 2 * sizeof(double));

            cudaStatus = cudaMemcpy(d_inp, addr, l * 4 * sizeof(int), cudaMemcpyHostToDevice);
            cudaStatus = cudaMemcpy(d_Es, Es, l * 2 * sizeof(double), cudaMemcpyHostToDevice);

            numblock = ceil((double)x*y*z / threadsPerBlock);
            cudaStatus = cudaMemcpy(d_sum1, sum1, l * sizeof(double), cudaMemcpyHostToDevice);
            cudaStatus = cudaMemcpy(d_sum2, sum2, x * y * z * sizeof(double), cudaMemcpyHostToDevice);

            //cout << "Forward Projection..." << endl;
            forward_proj << <numblock, threadsPerBlock >> > (d_sum1, d_image_u, d_CS, d_inp, d_Es, l, x, y, z, xsize, ysize, zsize, E_inc, scaxy, absdoi, scahighlim, scalowlim);

            cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            }
            cudaStatus = cudaDeviceSynchronize();
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
            }
            //cudaStatus = cudaMemcpy(sum1, d_sum1, l * sizeof(double), cudaMemcpyDeviceToHost);

            //for (int i = 0; i < l; i++)
                //cout << sum1[i] << endl;

            //void backward_proj(double* sum2, double* sum1, int* d_inpu, double* doi, double* beta, int inputsize, int x, int y, int z, double xsize, double ysize, double zsize, double E_inc)

            //cout << "Backward Projection..." << endl;


            backward_proj << <numblock, threadsPerBlock >> > (d_sum2, d_sum1, d_CS, d_inp, d_Es, l, x, y, z, xsize, ysize, zsize, E_inc, scaxy, absdoi, scahighlim, scalowlim);
            cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            }
            cudaStatus = cudaDeviceSynchronize();
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
            }

            numblock = ceil((double)x * y * z / threadsPerBlock);
            normal << <numblock, threadsPerBlock >> > (d_image_u, d_image_t, sp, d_sum2, x, y, z, subset);
            cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            }
            cudaStatus = cudaDeviceSynchronize();
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
            }

            cudaFree(d_sum1);
            cudaFree(d_sum2);
            cudaFree(d_inp);
            cudaFree(d_Es);
            
        }
        
        cudaStatus = cudaMemcpy(image, d_image_u, x * y * z * sizeof(double), cudaMemcpyDeviceToHost);

        string num = to_string(iter + 1);
        ofstream outp(foldername+"/" + num + ".txt");
        //ofstream sum2p("C:\\kim\\DOICCLMMLEM\\Output\\" + num + "sum2.txt");
        //cudaStatus = cudaMemcpy(sum2check, d_sum2, x * y * z * sizeof(double), cudaMemcpyDeviceToHost);
        double max = 0;
        
        for (int i = 0; i < x * y * z; i++)
        {
            if (max < image[i])
                max = image[i];
        }

        for (int i = 0; i < x * y * z; i++)
        {
            image[i] = image[i] / max;
        }

        for (int k = 0; k < z; k++)
        {
            for (int j = 0; j < y; j++)
            {
                for (int i = 0; i < x; i++)
                {
                    if (pow(i - x / 2 + 0.5, 2) + pow(j - y / 2 + 0.5, 2) > pow(y / 2 - 0, 2))
                        outp << 0<< "\t";
                    else
                        outp << image[i + x * (j + y * k)] << "\t";

                }
                outp << endl;
                
            }
        }
        

        cudaStatus = cudaMemcpy(d_image_t, image, x * y * z * sizeof(double), cudaMemcpyHostToDevice);
        cudaStatus = cudaMemcpy(d_image_u, image, x * y * z * sizeof(double), cudaMemcpyHostToDevice);


        cout << iter + 1 << " iterations complete" << endl;

        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

        std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::seconds>(end - begin).count() << "[s]" << std::endl << endl;
    }



}

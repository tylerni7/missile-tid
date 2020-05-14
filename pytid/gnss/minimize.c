// gcc -march=native -O3 minimize.c -shared -fPIC -o brute.so

#define SEARCH_SIZE 300

double brute_force(int ddn1, int ddn2, int *ws, double l1, double l2, double *Bis, int *n1s, int *n2s) {

    int n1_11,n1_12,n1_21,n1_22;
    int n2_11,n2_12,n2_21,n2_22;

    double min = 10000;

    double err, tmp;

    for (n1_11 = -SEARCH_SIZE; n1_11 < SEARCH_SIZE; n1_11++) {
        n2_11 = n1_11 - ws[0];
        for (n1_12 = -SEARCH_SIZE; n1_12 < SEARCH_SIZE; n1_12++) {
            n2_12 = n1_12 - ws[1];
            for (n1_21 = -SEARCH_SIZE; n1_21 < SEARCH_SIZE; n1_21++) {
                n2_21 = n1_21 - ws[2];


                n1_22 = ddn1 - n1_11 + n1_12 + n1_21;
                n2_22 = n1_22 - ws[3];

                err = 0;
                tmp = l1 * n1_11 - l2 * n2_11 - Bis[0];
                err += tmp*tmp;
                tmp = l1 * n1_12 - l2 * n2_12 - Bis[1];
                err += tmp*tmp;
                tmp = l1 * n1_21 - l2 * n2_21 - Bis[2];
                err += tmp*tmp;
                tmp = l1 * n1_22 - l2 * n2_22 - Bis[3];
                err += tmp*tmp;
                
                if (err < min) {
                //    printf("new min %0.2f\n", min);
                    min = err;
                    n1s[0] = n1_11;
                    n1s[1] = n1_12;
                    n1s[2] = n1_21;
                    n1s[3] = n1_22;
                    n2s[0] = n2_11;
                    n2s[1] = n2_12;
                    n2s[2] = n2_21;
                    n2s[3] = n2_22;
                }
            }
        }
    }
    return min;
}

double brute_force_harder(int *ws, double l1, double l2, double *Bis, int *n1s, int *n2s) {
    // don't use ddn1 and ddn2

    int n1_11,n1_12,n1_21,n1_22;
    int n2_11,n2_12,n2_21,n2_22;

    double min = 10000;

    double err, tmp;

    for (n1_11 = -SEARCH_SIZE; n1_11 < SEARCH_SIZE; n1_11++) {
        n2_11 = n1_11 - ws[0];
        for (n1_12 = -SEARCH_SIZE; n1_12 < SEARCH_SIZE; n1_12++) {
            n2_12 = n1_12 - ws[1];
            for (n1_21 = -SEARCH_SIZE; n1_21 < SEARCH_SIZE; n1_21++) {
                n2_21 = n1_21 - ws[2];

                for (n1_22 = -SEARCH_SIZE; n1_22 < SEARCH_SIZE; n1_22++) {
                    n2_22 = n1_22 - ws[3];

                    err = 0;
                    tmp = l1 * n1_11 - l2 * n2_11 - Bis[0];
                    err += tmp*tmp;
                    tmp = l1 * n1_12 - l2 * n2_12 - Bis[1];
                    err += tmp*tmp;
                    tmp = l1 * n1_21 - l2 * n2_21 - Bis[2];
                    err += tmp*tmp;
                    tmp = l1 * n1_22 - l2 * n2_22 - Bis[3];
                    err += tmp*tmp;
                    
                    if (err < min) {
                    //    printf("new min %0.2f\n", min);
                        min = err;
                        n1s[0] = n1_11;
                        n1s[1] = n1_12;
                        n1s[2] = n1_21;
                        n1s[3] = n1_22;
                        n2s[0] = n2_11;
                        n2s[1] = n2_12;
                        n2s[2] = n2_21;
                        n2s[3] = n2_22;
                    }
                }
            }
        }
    }
    return min;
}

double brute_force_dd2(int dd, double wavelength, double *Bs, int *ns) {
    int n1, n2, n3, n4;
    double min = 10000;
    double err, tmp;

    for (n1 = -SEARCH_SIZE; n1 < SEARCH_SIZE; n1++) {
        for (n2 = -SEARCH_SIZE; n2 < SEARCH_SIZE; n2++) {
            for (n3 = -SEARCH_SIZE; n3 < SEARCH_SIZE; n3++) {
                n4 = dd - n1 + n2 + n3;

                err = 0;
                tmp = wavelength * n1 - Bs[0];
                err += tmp*tmp;
                tmp = wavelength * n2 - Bs[1];
                err += tmp*tmp;
                tmp = wavelength * n3 - Bs[2];
                err += tmp*tmp;
                tmp = wavelength * n4 - Bs[3];
                err += tmp*tmp;

                if (err < min) {
                    min = err;
                    ns[0] = n1;
                    ns[1] = n2;
                    ns[2] = n3;
                    ns[3] = n4;
                }

            }
        }
    }
    return min;
}

double brute_force_dd(int dd, double wavelength, double *Bs, int *ns) {
    int in1, in2, in3;
    int n1, n2, n3, n4;
    double min = 10000;
    double err, tmp;

    for (in1 = 1; in1 < 2*SEARCH_SIZE; in1++) {
        // search 0, 1, -1, 2, -2, 3, -3...
        n1 = ((in1&1)?-1:1) * (in1 >> 1);
        for (in2 = 1; in2 < 2*SEARCH_SIZE; in2++) {
            n2 = ((in2&1)?-1:1) * (in2 >> 1);
            for (in3 = 1; in3 < 2*SEARCH_SIZE; in3++) {
                n3 = ((in3&1)?-1:1) * (in3 >> 1);
                n4 = dd - n1 + n2 + n3;

                err = 0;
                tmp = wavelength * n1 - Bs[0];
                err += tmp*tmp;
                tmp = wavelength * n2 - Bs[1];
                err += tmp*tmp;
                tmp = wavelength * n3 - Bs[2];
                err += tmp*tmp;
                tmp = wavelength * n4 - Bs[3];
                err += tmp*tmp;

                if (err < min) {
                    min = err;
                    ns[0] = n1;
                    ns[1] = n2;
                    ns[2] = n3;
                    ns[3] = n4;
                    // stop if we get close enough
                    if (err < 0.01) {
                        return err;
                    }
                }

            }
        }
    }
    return min;
}

//Alexandros Kokkinos 4084, Euaggelos Tempelopoulos 4175, Aggeliki Gkavardina 4042
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void populate(FILE *file, int iterations, double x_min, double x_max, double y_min, double y_max);

int main() {
    FILE *file = fopen("examples", "w+");
    if (file == NULL) {
        perror("Opening Examples: FAILED");
        return EXIT_FAILURE;
    }

    populate(file, 100, -2.0, -1.6, 1.6, 2.0);
    populate(file, 100, -1.2, -0.8, 1.6, 2.0);
    populate(file, 100, -0.4, 0.0, 1.6, 2.0);
    populate(file, 100, -1.8, -1.4, 0.8, 1.2);
    populate(file, 100, -0.6, -0.2, 0.8, 1.2);
    populate(file, 100, -2.0, -1.6, 0.0, 0.4);
    populate(file, 100, -1.2, -0.8, 0.0, 0.4);
    populate(file, 100, -0.4, 0.0, 0.0, 0.4);

    populate(file, 200, -2.0, 0.0, 0.0, 2.0);

    fclose(file);
    return 0;
}

// Create random points based on given range and for given iterations
void populate(FILE *file, int iterations, double x_min, double x_max, double y_min, double y_max) {
    int i;
    double x_range = fabs(x_max - x_min);
    double y_range = fabs(y_max - y_min);

    for (i = 0; i < iterations; i++) {
        double x = x_min + ((double)rand() / RAND_MAX) * x_range;
        double y = y_min + ((double)rand() / RAND_MAX) * y_range;
        fprintf(file, "%f %f\n", x, y);
    }
}
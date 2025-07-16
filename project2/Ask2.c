//Alexandros Kokkinos 4084, Euaggelos Tempelopoulos 4175, Aggeliki Gkavardina 4042
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h> 

#define MAX_POINTS 1000
#define MAX_ITER 20

typedef struct {
    double x, y;
} Point;

// Calculate Distance to be used as error severity
double distance(Point a, Point b) {
    return sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2));
}

int main() {
    // Load the examples to be used by the k-means algo.
    FILE *file = fopen("examples", "r");
    if (file == NULL) {
        perror("Opening examples: FAILED");
        return 1;
    }

    Point points[MAX_POINTS];
    int num_points = 0;

    // Loading each point from the examples file
    while (fscanf(file, "%lf %lf", &points[num_points].x, &points[num_points].y) == 2) {
        num_points++;
    }
    fclose(file);

    int M_values[] = {4, 6, 8, 10, 12};
    int num_M_values = sizeof(M_values) / sizeof(M_values[0]);

    // Create CSV file for examples and best centers
    FILE *plot_file = fopen("plot_data.csv", "w"); 
    if (plot_file == NULL) {
        perror("Error opening plot file");
        return 1;
    }

    // Create CSV file for error value per M value
    FILE *error_file = fopen("error_data.csv", "w"); 
    if (error_file == NULL) {
        perror("Error opening error file");
        return 1;
    }

    fprintf(error_file, "M,Error\n"); 

    // Create a third column to distinguish examples from centers
    fprintf(plot_file, "X,Y,Type\n"); 
    for (int p = 0; p < num_points; p++) {
        fprintf(plot_file, "%f,%f,+\n", points[p].x, points[p].y); 
    }

    for (int i = 0; i < num_M_values; i++) {
        int M = M_values[i];
        double min_error = INFINITY;
        Point best_centers[M];

        printf("M = %d:\n", M);

        for (int j = 0; j < MAX_ITER; j++) {
            Point centers[M];
            int assignments[MAX_POINTS];

            // Initialize random centers
            for (int k = 0; k < M; k++) {
                centers[k] = points[rand() % num_points];
            }

            // Run k-means
            for (int iter = 0; iter < 100; iter++) {
                // Assign points to teams
                for (int p = 0; p < num_points; p++) {
                    double min_dist = INFINITY;
                    int closest_center = 0;
                    for (int c = 0; c < M; c++) {
                        double dist = distance(points[p], centers[c]);
                        if (dist < min_dist) {
                            min_dist = dist;
                            closest_center = c;
                        }
                    }
                    assignments[p] = closest_center;
                }

                // Updating centers
                for (int c = 0; c < M; c++) {
                    centers[c].x = 0.0;
                    centers[c].y = 0.0;
                    int count = 0;
                    for (int p = 0; p < num_points; p++) {
                        if (assignments[p] == c) {
                            centers[c].x += points[p].x;
                            centers[c].y += points[p].y;
                            count++;
                        }
                    }
                    if (count > 0) {
                        centers[c].x /= count;
                        centers[c].y /= count;
                    }
                }
            }

            // Error valuation
            double error = 0.0;
            for (int p = 0; p < num_points; p++) {
                error += distance(points[p], centers[assignments[p]]);
            }

            printf("Iteration %d: Error = %f, Centers: ", j + 1, error);
            for (int k = 0; k < M; k++) {
                printf("(%f, %f) ", centers[k].x, centers[k].y);
            }
            printf("\n"); 

            if (error < min_error) {
                min_error = error;
                for (int k = 0; k < M; k++) {
                    best_centers[k] = centers[k];
                }
            }
        }

        printf("\nBest Iteration for M = %d:\n", M);
        printf("Error: %f\n", min_error);
        fprintf(error_file, "%d,%f\n", M, min_error); // Store error value for each M value

        printf("Centers:\n");
        for (int k = 0; k < M; k++) {
            printf("(%f, %f)\n", best_centers[k].x, best_centers[k].y);
            fprintf(plot_file, "%f,%f,*\n", best_centers[k].x, best_centers[k].y); // Assign "*" Type for centers
        }

        printf("\n");
    }

    fclose(plot_file);
    fclose(error_file);

    return 0;
}
#include <stdio.h>

#include <assert.h>
#include <string.h>
#include <time.h>
#include <mpi.h>

#include "auxiliar.h"

/// TODO
/// Reading the planes from a file for MPI
void read_planes_mpi(const char* filename, PlaneList* planes, int* N, int* M, double* x_max, double* y_max, int rank, int size, int* tile_displacements)
{
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Error opening file");
        return;
    }

    char line[MAX_LINE_LENGTH];
    int num_planes = 0;

    // Reading header
    fgets(line, sizeof(line), file);
    fgets(line, sizeof(line), file);
    sscanf(line, "# Map: %lf, %lf : %d %d", x_max, y_max, N, M);
    fgets(line, sizeof(line), file);
    sscanf(line, "# Number of Planes: %d", &num_planes);
    fgets(line, sizeof(line), file);

    int total_tiles = (*N) * (*M);
    for (int i = 0; i < size; i++) {
        tile_displacements[i] = (total_tiles / size) * i;
    }

    // Reading plane data
    int nplanes_inserted_local = 0;
    while (fgets(line, sizeof(line), file)) {
        int idx;
        double x, y, vx, vy;
        if (sscanf(line, "%d %lf %lf %lf %lf",
                   &idx, &x, &y, &vx, &vy) == 5) {
            int index_i = get_index_i(x, *x_max, *N);
            int index_j = get_index_j(y, *y_max, *M);
            int index_map = get_index(index_i, index_j, *N, *M);
            int rank_plane = get_rank_from_indices(index_i, index_j, *N, *M, tile_displacements, size);
            if (rank_plane != rank) {
                continue; // Skip planes that don't belong to this rank
            }
            insert_plane(planes, idx, index_map, rank, x, y, vx, vy);
            nplanes_inserted_local++;
        }
    }
    fclose(file);
    int nplanes_inserted_global = 0;
    MPI_Allreduce(&nplanes_inserted_local, &nplanes_inserted_global, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    printf("Total planes read: %d\n", nplanes_inserted_local);
    printf("Total planes read: %d\n", nplanes_inserted_global);
    assert(num_planes == nplanes_inserted_global);
}

/// TODO
/// Communicate planes using mainly Send/Recv calls with default data types
void communicate_planes_send(PlaneList* list, int N, int M, double x_max, double y_max, int rank, int size, int* tile_displacements)
{   
    PlaneNode* current = list->head;
    int* to_send = (int*) malloc(sizeof(int) * size);
    while (current != NULL) {
        //Calcular cuantos aviones tengo que comunicar
        int index_i = get_index_i(current->x, x_max, N);
        int index_j = get_index_j(current->y, y_max, M);
        int plane_rank = get_rank_from_indices(index_i, index_j, N, M, tile_displacements, size);
        
        if (plane_rank != rank){
            to_send[plane_rank] += 1;
        }

        current = current->next;
    }

    //comunicar cuantos aviones a cada rank con un alltoall
    int* to_receive = (int*) malloc(sizeof(int) * size);
    MPI_Alltoall(to_send, 1, MPI_INT, to_receive, 1, MPI_INT, MPI_COMM_WORLD);
    int total_to_send = 0;
    for (int i = 0; i < size; i++) {
        total_to_send += to_send[i];
    }
    MPI_Request req[total_to_send];

    //enviar los aviones
    current = list->head;
    while(current != NULL){
        int index_i = get_index_i(current->x, x_max, N);
        int index_j = get_index_j(current->y, y_max, M);
        int index_map = get_index(index_i, index_j, *N, *M);
        int plane_rank = get_rank_from_indices(index_i, index_j, N, M, tile_displacements, size);

        //cast index_map to double
        double index_map_d = (double) index_map;

        double* plane_info = (double*) malloc(sizeof(double) * 5);

        plane_info[0] = (double) current->index_plane;
        plane_info[1] = (double) current->x;
        plane_info[2] = (double) current->y;
        plane_info[3] = (double) current->vx;
        plane_info[4] = (double) current->vy;
        

        if (plane_rank != rank){
            MPI_Isend( plane_info, 5 , MPI_DOUBLE, plane_rank, 0 , MPI_COMM_WORLD, req);
        }
        current = current->next;
    }

    //enviar 

}

/// TODO
/// Communicate planes using all to all calls with default data types
void communicate_planes_alltoall(PlaneList* list, int N, int M, double x_max, double y_max, int rank, int size, int* tile_displacements)
{
 
}

typedef struct {
    int    index_plane;
    double x;
    double y;
    double vx;
    double vy;
} MinPlaneToSend;

/// TODO
/// Communicate planes using all to all calls with custom data types
void communicate_planes_struct_mpi(PlaneList* list, int N, int M, double x_max, double y_max, int rank, int size, int* tile_displacements)
{

}

int main(int argc, char **argv) {
    int debug = 0;                      // 0: no debug, 1: shows all planes information during checking
    int N = 0, M = 0;                   // Grid dimensions
    double x_max = 0.0, y_max = 0.0;    // Total grid size
    int max_steps;                      // Total simulation steps
    char* input_file;                   // Input file name
    int check;                          // 0: no check, 1: check the simulation is correct

    int rank, size;

    /// TODO
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int tile_displacements[size+1];
    int mode = 0;
    if (argc == 5) {
        input_file = argv[1];
        max_steps = atoi(argv[2]);
        if (max_steps <= 0) {
            fprintf(stderr, "max_steps needs to be a positive integer\n");
            return 1;
        }
        mode = atoi(argv[3]);
        if (mode > 2 || mode < 0) {
            fprintf(stderr, "mode needs to be a value between 0 and 2\n");
            return 1;
        }
        check = atoi(argv[4]);
        if (check >= 2 || check < 0) {
            fprintf(stderr, "check needs to be a 0 or 1\n");
            return 1;
        }
    }
    else {
        fprintf(stderr, "Usage: %s <filename> <max_steps> <mode> <check>\n", argv[0]);
        return 1;
    }

    PlaneList owning_planes = {NULL, NULL};
    read_planes_mpi(input_file, &owning_planes, &N, &M, &x_max, &y_max, rank, size, tile_displacements);
    PlaneList owning_planes_t0 = copy_plane_list(&owning_planes);

    //print_planes_par_debug(&owning_planes);

    double time_sim = 0., time_comm = 0, time_total=0;

    double start_time = MPI_Wtime();
    int step = 0;
    for (step = 1; step <= max_steps; step++) {
        double start = MPI_Wtime();
        PlaneNode* current = owning_planes.head;
        while (current != NULL) {
            current->x += current->vx;
            current->y += current->vy;
            current = current->next;
        }
        filter_planes(&owning_planes, x_max, y_max);
        time_sim += MPI_Wtime() - start;

        start = MPI_Wtime();
        if (mode == 0)
            communicate_planes_send(&owning_planes, N, M, x_max, y_max, rank, size, tile_displacements);
        else if (mode == 1)
            communicate_planes_alltoall(&owning_planes, N, M, x_max, y_max, rank, size, tile_displacements);
        else
            communicate_planes_struct_mpi(&owning_planes, N, M, x_max, y_max, rank, size, tile_displacements);
        time_comm += MPI_Wtime() - start;
    }
    time_total = MPI_Wtime() - start_time;

    /// TODO Check computational times


    if (rank == 0) {
        printf("Flight controller simulation: #input %s mode: %d size: %d\n", input_file, mode, size);
        printf("Time simulation:     %.2fs\n", time_sim);
        printf("Time communication:  %.2fs\n", time_comm);
        printf("Time total:          %.2fs\n", time_total);
    }

    if (check ==1)
        check_planes_mpi(&owning_planes_t0, &owning_planes, N, M, x_max, y_max, max_steps, tile_displacements, size, debug);

    MPI_Finalize();
    return 0;
}

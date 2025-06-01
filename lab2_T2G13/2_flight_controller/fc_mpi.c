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
    printf("Total planes read by rank %d: %d\n", rank, nplanes_inserted_local);
    printf("Total planes read globally: %d\n", nplanes_inserted_global);
    assert(num_planes == nplanes_inserted_global);
}

/// TODO
/// Communicate planes using mainly Send/Recv calls with default data types
void communicate_planes_send(PlaneList* list, int N, int M, double x_max, double y_max, int rank, int size, int* tile_displacements)
{   
    int* number_of_planes_to_send = (int*) calloc(size, sizeof(int));
    PlaneNode* current = list->head;
    while (current != NULL) {
        //Calcular cuantos aviones tengo que comunicar
        int index_i = get_index_i(current->x, x_max, N);
        int index_j = get_index_j(current->y, y_max, M);
        int plane_rank = get_rank_from_indices(index_i, index_j, N, M, tile_displacements, size);
        
        if (plane_rank != rank){
            number_of_planes_to_send[plane_rank] += 1;
        }

        current = current->next;
    }

    //comunicar cuantos aviones a cada rank con un alltoall
    int* number_of_planes_to_receive = (int*) calloc(size, sizeof(int));
    MPI_Alltoall(number_of_planes_to_send, 1, MPI_INT, number_of_planes_to_receive, 1, MPI_INT, MPI_COMM_WORLD);

    // comunicar cuantos aviones enviaremos en total
    int total_planes_to_send = 0;
    for (int i = 0; i < size; i++) {
        total_planes_to_send += number_of_planes_to_send[i];
    }
    
    // declaramos el buffer y requests para enviar los aviones
    MPI_Request req[total_planes_to_send];
    double* planes_to_send = (double*) malloc(total_planes_to_send * 5 * sizeof(double));
    
    //enviar los aviones
    current = list->head;
    int req_index = 0;
    while(current != NULL){
        PlaneNode* next = current->next;
        int index_i = get_index_i(current->x, x_max, N);
        int index_j = get_index_j(current->y, y_max, M);
        int plane_rank = get_rank_from_indices(index_i, index_j, N, M, tile_displacements, size);

        if (plane_rank != rank){
            planes_to_send[req_index * 5] = (double) current->index_plane;;
            planes_to_send[req_index * 5 + 1] = current->x;
            planes_to_send[req_index * 5 + 2] = current->y;
            planes_to_send[req_index * 5 + 3] = current->vx;
            planes_to_send[req_index * 5 + 4] = current->vy;
            MPI_Isend( &planes_to_send[req_index * 5], 5 , MPI_DOUBLE, plane_rank, 0 , MPI_COMM_WORLD, &req[req_index]);
            remove_plane(list, current);
            req_index++;
        }
        current = next;
    }

    //recibir los aviones
    for (int i = 0; i < size; i++){
        for (int j = 0; j < number_of_planes_to_receive[i]; j++){
            double plane_buffer[5];
            MPI_Recv(plane_buffer, 5, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            //calculate index_map
            int index_i = get_index_i(plane_buffer[1], x_max, N);
            int index_j = get_index_j(plane_buffer[2], y_max, M);
            int index_map = get_index(index_i, index_j, N, M);

            insert_plane(list, plane_buffer[0], index_map , rank, plane_buffer[1], plane_buffer[2], plane_buffer[3], plane_buffer[4]);
        }        
    }
    

    MPI_Waitall(total_planes_to_send, req, MPI_STATUSES_IGNORE);
    free(planes_to_send);
    free(number_of_planes_to_receive);
    free(number_of_planes_to_send);
}

/// TODO
/// Communicate planes using all to all calls with default data types
void communicate_planes_alltoall(PlaneList* list, int N, int M, double x_max, double y_max, int rank, int size, int* tile_displacements)
{   
    int* send_counts = (int*) calloc(size, sizeof(int));
    PlaneNode* current = list->head;
    while (current != NULL) {
        int index_i = get_index_i(current->x, x_max, N);
        int index_j = get_index_j(current->y, y_max, M);
        int plane_rank = get_rank_from_indices(index_i, index_j, N, M, tile_displacements, size);
        if (plane_rank != rank) {
            send_counts[plane_rank]++;
        }
        current = current->next;
    }

    int* recv_counts = (int*) calloc(size, sizeof(int));
    MPI_Alltoall(send_counts, 1, MPI_INT, recv_counts, 1, MPI_INT, MPI_COMM_WORLD);

    int total_number_planes_to_send = 0;
    int total_number_planes_to_recv = 0;

    for (int i = 0; i < size; i++) {
        total_number_planes_to_send += send_counts[i];
        total_number_planes_to_recv += recv_counts[i];
    }

    double* send_buffer = (double*) malloc(total_number_planes_to_send * 5 * sizeof(double));
    int send_idx = 0;
    current = list->head;
    while (current != NULL) {
        PlaneNode* next = current->next;
        int index_i = get_index_i(current->x, x_max, N);
        int index_j = get_index_j(current->y, y_max, M);
        int plane_rank = get_rank_from_indices(index_i, index_j, N, M, tile_displacements, size);
        
        if (plane_rank != rank) {
            send_buffer[send_idx*5] = (double)current->index_plane;
            send_buffer[send_idx*5 + 1] = current->x;
            send_buffer[send_idx*5 + 2] = current->y;
            send_buffer[send_idx*5 + 3] = current->vx;
            send_buffer[send_idx*5 + 4] = current->vy;
            
            remove_plane(list, current);
            send_idx++;
        } 
        current = next;
    }

    double* recv_buffer = (double*) malloc(total_number_planes_to_recv * 5 * sizeof(double));

    int* send_displacements = malloc(size * sizeof(int));
    send_displacements[0] = 0;
    for (int i = 1; i < size; i++) {
        send_displacements[i] = send_displacements[i-1] + (send_counts[i-1]*5);
    }
    int* recv_displacements = malloc(size * sizeof(int));
    recv_displacements[0] = 0;
    for (int i = 1; i < size; i++) {
        recv_displacements[i] = recv_displacements[i-1] + (recv_counts[i-1]*5);
    }
    MPI_Alltoallv(send_buffer, total_number_planes_to_send, send_displacements, MPI_DOUBLE, recv_buffer, total_number_planes_to_recv, recv_displacements, MPI_DOUBLE, MPI_COMM_WORLD);
    // Process received planes
    for (int i = 0; i < total_number_planes_to_recv; i += 5) {
        int index_plane = (int)recv_buffer[i];
        double x = recv_buffer[i + 1];
        double y = recv_buffer[i + 2];
        double vx = recv_buffer[i + 3];
        double vy = recv_buffer[i + 4];
        int index_i = get_index_i(x, x_max, N);
        int index_j = get_index_j(y, y_max, M);
        int index_map = get_index(index_i, index_j, N, M);
        insert_plane(list, index_plane, index_map, rank, x, y, vx, vy);
    }

    free(send_counts);
    free(recv_counts);
    free(send_displacements);
    free(recv_displacements);
    free(send_buffer);
    free(recv_buffer);
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

    MPI_Datatype planeType, oldtypes[2];
    int blockcounts[2];
    MPI_Aint offsets[2], extent, lb;

    offsets[0] = 0;
    oldtypes[0] = MPI_INT;
    blockcounts[0] = 1;

    MPI_Type_get_extent(MPI_INT, &lb, &extent);
    offsets[1] = extent;
    oldtypes[1] = MPI_DOUBLE;
    blockcounts[1] = 4;

    MPI_Type_create_struct(2, blockcounts, offsets, oldtypes, &planeType);
    MPI_Type_commit(&planeType);

    
    int* to_send = (int*) calloc(size, sizeof(int));
    PlaneNode* current = list->head;
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
    free(to_send);
    MPI_Request req[total_to_send];
    MinPlaneToSend planes_information[total_to_send];
    //enviar los aviones
    current = list->head;
    int req_index = 0;
    while(current != NULL){
        int index_i = get_index_i(current->x, x_max, N);
        int index_j = get_index_j(current->y, y_max, M);
        int plane_rank = get_rank_from_indices(index_i, index_j, N, M, tile_displacements, size);

        if (plane_rank != rank){
            MinPlaneToSend plane_to_send = {current->index_plane, current->x, current->y, current->vx, current->vy};
            planes_information[req_index] = plane_to_send;
            MPI_Isend( &planes_information[req_index], 1 , planeType, plane_rank, 0 , MPI_COMM_WORLD, &req[req_index]);
            remove_plane(list, current);
            req_index++;
        }
        current = current->next;
    }

    //recibir los aviones
    for (int i = 0; i < size; i++){
        for (int j = 0; j < to_receive[i]; j++){
            MinPlaneToSend plane_to_receive;
            MPI_Recv( &plane_to_receive, 1, planeType, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            insert_plane(list, plane_to_receive.index_plane, plane_to_receive.index_plane, rank, plane_to_receive.x, plane_to_receive.y, plane_to_receive.vx, plane_to_receive.vy);
        }        
    }
    free(to_receive);

    MPI_Waitall(total_to_send, req, MPI_STATUSES_IGNORE);
}

int main(int argc, char **argv) {
    int debug = 1;                      // 0: no debug, 1: shows all planes information during checking
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

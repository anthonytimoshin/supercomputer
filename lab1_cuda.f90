program lab1
    use iso_fortran_env, only: real64
    use cudafor
    implicit none
    real(real64) :: start_time, end_time, total_time
    ! инициализация переменных, объявление констант
    integer, parameter :: elements_edu = 10000, elements_test = 100, fields = 5
    integer, parameter :: max_floors = 50, max_rooms = 5, max_square = 300, distincts = 3
    integer :: i, j, k ! итераторы для циклов
    real :: f, r, s, d
    integer :: apartments(elements_edu, fields)
    integer :: test_apartments(elements_test, fields)
    real, parameter :: base_price_per_sqm = 100.00
    real, parameter :: floorK = 0.2
    real, parameter :: roomK = 0.8
    real, parameter :: distinctK(3) = [0.6, 1.0, 1.8]
    integer :: k1 = max_square / max_floors, k2 = max_square / max_rooms
    integer:: k3 = 1, k4 = max_square / distincts
    integer :: KNN = 35
    integer :: dist_temp, id_temp
    integer :: euclidean_distance(elements_edu)
    integer :: manhattan_distance(elements_edu)
    integer :: id_euclidean(elements_edu)
    integer :: id_manhattan(elements_edu)
    integer :: euclidean_predict_price, manhattan_predict_price
    real :: euclidean_accuracy(elements_test), manhattan_accuracy(elements_test)
    integer, device, allocatable :: apartments_d(:,:)
    integer, device :: test_apart(4)
    integer, device, allocatable :: euclidean_distance_d(:)
    integer, device, allocatable :: manhattan_distance_d(:)
    integer, device, allocatable :: id_euclidean_d(:)
    integer, device, allocatable :: id_manhattan_d(:)
    type(dim3) :: threads, blocks
    integer :: istat
    call cpu_time(start_time)
    ! генерация обучающей выборки
    call random_seed()
   
    do i = 1, elements_edu
        call random_number(f)
        apartments(i, 1) = int(f * max_floors + 1)
        call random_number(r)
        apartments(i, 2) = int(r * max_rooms + 1)
        call random_number(s)
        apartments(i, 3) = int(s * max_square + apartments(i, 2) * 10)
        call random_number(d)
        apartments(i, 4) = int(d * distincts + 1)
        apartments(i, 5) = int(apartments(i, 1) * floorK * apartments(i, 2) * roomK * &
                           apartments(i, 3) * base_price_per_sqm * distinctK(apartments(i, 4)))
    end do
    print *, 'Обучающая выборка:'
    print *, ' Этаж', ' Кол-во комнат', ' Площадь', ' Район ID', ' Стоимость'
    do i = 1, 10
        print *, apartments(i, :)
    end do
    ! генерация тестовой выборки
    do i = 1, elements_test
        call random_number(f)
        test_apartments(i, 1) = int(f * max_floors + 1)
        call random_number(r)
        test_apartments(i, 2) = int(r * max_rooms + 1)
        call random_number(s)
        test_apartments(i, 3) = int(s * max_square + test_apartments(i, 2) * 10)
        call random_number(d)
        test_apartments(i, 4) = int(d * distincts + 1)
        test_apartments(i, 5) = int(test_apartments(i, 1) * floorK * test_apartments(i, 2) * roomK * &
                                test_apartments(i, 3) * base_price_per_sqm * distinctK(test_apartments(i, 4)))
    end do

    ! Allocate device memory
    allocate(apartments_d(elements_edu, fields))
    allocate(euclidean_distance_d(elements_edu))
    allocate(manhattan_distance_d(elements_edu))
    allocate(id_euclidean_d(elements_edu))
    allocate(id_manhattan_d(elements_edu))

    ! Copy training data to device
    apartments_d = apartments

    ! Set grid and block sizes
    threads = dim3(256, 1, 1)
    blocks = dim3((elements_edu + threads%x - 1) / threads%x, 1, 1)

    ! === ОДНОПОТОЧНАЯ РЕАЛИЗАЦИЯ АЛГОРИТМА KNN НА CPU, НО С ПАРАЛЛЕЛЬНЫМ ВЫЧИСЛЕНИЕМ РАССТОЯНИЙ НА GPU ===
   
    do i = 1, elements_test
        ! Copy test apartment features to device
        test_apart = test_apartments(i, 1:4)

        ! Launch kernel to compute distances on GPU
        call compute_distances<<<blocks, threads>>>(apartments_d, test_apart, euclidean_distance_d, manhattan_distance_d, &
                                                    id_euclidean_d, id_manhattan_d, k1, k2, k3, k4, elements_edu)
        istat = cudaDeviceSynchronize()

        ! Copy distances and ids back to host
        euclidean_distance = euclidean_distance_d
        manhattan_distance = manhattan_distance_d
        id_euclidean = id_euclidean_d
        id_manhattan = id_manhattan_d

        ! пузырьковая сортировка по расстоянию на host
        do j = 1, elements_edu - 1
            do k = j + 1, elements_edu
                if (euclidean_distance(j) > euclidean_distance(k)) then
                    dist_temp = euclidean_distance(j)
                    euclidean_distance(j) = euclidean_distance(k)
                    euclidean_distance(k) = dist_temp
                    id_temp = id_euclidean(j)
                    id_euclidean(j) = id_euclidean(k)
                    id_euclidean(k) = id_temp
                end if
                if (manhattan_distance(j) > manhattan_distance(k)) then
                    dist_temp = manhattan_distance(j)
                    manhattan_distance(j) = manhattan_distance(k)
                    manhattan_distance(k) = dist_temp
                    id_temp = id_manhattan(j)
                    id_manhattan(j) = id_manhattan(k)
                    id_manhattan(k) = id_temp
                end if
            end do
        end do
        euclidean_predict_price = 0
        manhattan_predict_price = 0
        do j = 1, KNN
            euclidean_predict_price = euclidean_predict_price + apartments(id_euclidean(j), 5)
            manhattan_predict_price = manhattan_predict_price + apartments(id_manhattan(j), 5)
        end do
        euclidean_predict_price = euclidean_predict_price / KNN
        manhattan_predict_price = manhattan_predict_price / KNN
        euclidean_accuracy(i) = real(euclidean_predict_price)/test_apartments(i, 5)
        manhattan_accuracy(i) = real(manhattan_predict_price)/test_apartments(i, 5)
        print *, 'Тестовая выборка:'
        print *, ' Этаж', ' Кол-во комнат', ' Площадь', ' Район ID',&
         ' Реал. цена', ' Евклид', ' Точность', ' Манхеттен', ' Точность'
        print *, test_apartments(i, :), euclidean_predict_price, euclidean_accuracy(i), &
                                        manhattan_predict_price, manhattan_accuracy(i)
    end do
    call cpu_time(end_time)
    total_time = end_time - start_time
    print *, 'Общее время выполнения: ', total_time, ' секунд'

    ! Deallocate device memory
    deallocate(apartments_d)
    deallocate(euclidean_distance_d)
    deallocate(manhattan_distance_d)
    deallocate(id_euclidean_d)
    deallocate(id_manhattan_d)

end program lab1

attributes(global) subroutine compute_distances(apart_d, test_a, eucl_d, manh_d, id_e, id_m, k1, k2, k3, k4, n)
    integer, value :: n, k1, k2, k3, k4
    integer, device :: apart_d(n, 5)
    integer, value :: test_a(4)
    integer, device :: eucl_d(n), manh_d(n), id_e(n), id_m(n)
    integer :: j
    real :: df1, df2, df3, df4
    j = (blockIdx%x - 1) * blockDim%x + threadIdx%x
    if (j <= n) then
        id_e(j) = j
        id_m(j) = j
        df1 = real(test_a(1) - apart_d(j, 1))
        df2 = real(test_a(2) - apart_d(j, 2))
        df3 = real(test_a(3) - apart_d(j, 3))
        df4 = real(test_a(4) - apart_d(j, 4))
        eucl_d(j) = int(sqrt(real(k1) * df1**2 + real(k2) * df2**2 + real(k3) * df3**2 + real(k4) * df4**2))
        manh_d(j) = int(real(k1) * abs(df1) + real(k2) * abs(df2) + real(k3) * abs(df3) + real(k4) * abs(df4))
    end if
end subroutine compute_distances

subroutine generate_dataset() ! генерация обучающей и тестовой выборок
    integer, parameter :: elements_edu = 100, elements_test = 1000, fields = 5
    integer, parameter :: max_floors = 50, max_rooms = 5, max_square = 300, distincts = 3
    integer :: i
    real :: f, r, s, d
    integer :: apartments(elements_edu, fields)
    integer :: test_apartments(elements_test, fields)
    real, parameter :: base_price_per_sqm = 50000.00
    real, parameter :: floorK = 0.2
    real, parameter :: roomK = 0.8
    real, parameter :: distinctK(3) = [0.6, 1.0, 1.8]

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
    print *, '      Этаж', '    Кол-во комнат', '   Площадь', '   Район ID', '  Стоимость'

    do i = 1, elements_edu
        print *, apartments(i, :)
    end do

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

    print *, 'Тестовая выборка:'
    print *, '       Этаж', '    Кол-во комнат', '   Площадь', '   Район ID', '  Реальная стоимость'

    do i = 1, elements_test - 900
        print *, test_apartments(i, :)
    end do

end subroutine generate_dataset

program lab1
    implicit none
    
    ! однопоточная реализация алгоритма KNN
    call random_seed()
    call generate_dataset()
    
end program lab1

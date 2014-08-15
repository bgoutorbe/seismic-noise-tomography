module utils

contains

    subroutine moving_avg(array, array_avg, n, window)
        ! function to perform a running avg
        implicit none
        integer :: n, window
        real, intent(in)  :: array(n)
		real, intent(out) :: array_avg(n)        
        integer :: i, imin, imax
       
        do i = 1, n           
            if (i<= window) then
                imin = 1
            else
                imin = i - window
            endif
            if (i >= n-window) then
                imax = n
            else
                imax = i + window
            endif

            array_avg(i) = sum(array(imin:imax)) / (imax-imin+1)
		enddo
    end subroutine
    
    !------------------------------------------------------------------------------------------------

    subroutine moving_avg_mask(array, array_avg, mask, n, window)
        ! function to perform a running avg on a masked array
        implicit none
        integer :: n, window
        real, intent(in)  :: array(n)
		real, intent(out) :: array_avg(n)  
        logical :: mask(n)
        
        integer :: i, imin, imax

        array_avg = 0.0        
        do i = 1, n
            if ( .not. mask(i) ) cycle            
            if (i<= window) then
                imin = 1
            else
                imin = i - window
            endif
            if (i >= n-window) then
                imax = n
            else
                imax = i + window
            endif

            array_avg(i) = sum(array(imin:imax),mask=mask(imin:imax)) / count(mask(imin:imax))
		enddo
    end subroutine

    !------------------------------------------------------------------------------------------------

end module

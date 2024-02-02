      subroutine nest(dnest,dp,np,nf) 
      double precision dp(np,nf), sp(np), sf(nf)
      double precision dnest, a, dnest_tmp     
      
      do i=1, np
        sp(i)=0.d0
	do j=1, nf
	  sp(i)=sp(i) + dp(i,j)
	enddo
      enddo
      
      do j=1, nf
        sf(j)=0.d0
	do i=1, np
	  sf(j)=sf(j) + dp(i,j)
	enddo 
      enddo
      
      dnest=0.d0
      ncount=0
      do i=1, np-1
        do j=i+1, np
	  a=0.d0
	  do k=1, nf
	    a=a+ dp(i,k)*dp(j,k)
	  enddo
	  ncount=ncount+1
	  if(sp(j)*sp(i).ne.0.d0) then
	    if(sp(j).lt.sp(i)) then
	      dnest=dnest + a/sp(j)
	    elseif(sp(j).gt.sp(i)) then
	      dnest=dnest + a/sp(i)
	    endif
	  endif
	enddo
      enddo
      
      dnest_tmp=0.d0
      ncount_tmp=0
      do i=1, nf-1
        do j=i+1, nf
	  a=0.d0
	  do k=1, np
	    a=a+ dp(k,i)*dp(k,j)
	  enddo
	  ncount=ncount+1
	  ncount_tmp=ncount_tmp+1
	  if(sf(j)*sf(i).ne.0.d0) then
	    if(sf(j).lt.sf(i)) then
	      dnest=dnest + a/sf(j)
	      dnest_tmp=dnest_tmp + a/sf(j)
	    elseif(sf(j).gt.sf(i)) then
	      dnest=dnest + a/sf(i)
	      dnest_tmp=dnest_tmp + a/sf(i)
	    endif
	  endif
	  
	enddo
      enddo
      dnest=dnest/dble(ncount)
 
      end

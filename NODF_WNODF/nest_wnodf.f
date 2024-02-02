      subroutine nest_w(dnest,dp,np,nf) 
      double precision dp(np,nf), sp(np), sf(nf)
      double precision deg_p(np), deg_f(nf)
      double precision dnest, a, dnest_tmp     
      
      do i=1, np
        sp(i)=0.d0
	deg_p(i)=0.d0
	do j=1, nf
	  sp(i)=sp(i) + dp(i,j)
	  if(dp(i,j).ne.0.d0) deg_p(i)=deg_p(i) + 1.d0
	enddo
      enddo
      
      do j=1, nf
        sf(j)=0.d0
	deg_f(j)=0.d0
	do i=1, np
	  sf(j)=sf(j) + dp(i,j)
	  if(dp(i,j).ne.0.d0) deg_f(j)=deg_f(j) + 1.d0
	enddo 
      enddo
      
      dnest=0.d0
      ncount=0
      do i=1, np-1
        do j=i+1, np
	  a=0.d0
	  if(deg_p(i)*deg_p(j).ne.0.d0) then	    
	  if(deg_p(i).lt.deg_p(j)) then	    
	    do k=1, nf
      if(dp(i,k).lt.dp(j,k).and.dp(i,k).ne.0.d0) a=a+ 1.d0
	    enddo
	    dnest=dnest + a/deg_p(i) 
	  elseif(deg_p(j).lt.deg_p(i)) then	    
	    do k=1, nf
      if(dp(j,k).lt.dp(i,k).and.dp(j,k).ne.0.d0) a=a+ 1.d0
	    enddo
	    dnest=dnest + a/deg_p(j) 
	  endif
	  endif
	  ncount=ncount+1
	enddo
      enddo
      
      dnest_tmp=0.d0
      ncount_tmp=0
      do i=1, nf-1
        do j=i+1, nf
	  a=0.d0
	  if(deg_f(i)*deg_f(j).ne.0.d0) then	    
	  if(deg_f(i).lt.deg_f(j)) then	    

	    do k=1, np
      if(dp(k,i).lt.dp(k,j).and.dp(k,i).ne.0.d0) a=a+1.d0
	    enddo
	    dnest=dnest + a/deg_f(i)
	    dnest_tmp=dnest_tmp + a/deg_f(i)
	  elseif(deg_f(j).lt.deg_f(i)) then	    
	  
	    do k=1, np
      if(dp(k,j).lt.dp(k,i).and.dp(k,j).ne.0.d0) a=a+1.d0
	    enddo
	    dnest=dnest + a/deg_f(j)
	    dnest_tmp=dnest_tmp + a/deg_f(j)
	  endif
	  endif
	  ncount_tmp=ncount_tmp+1
	  ncount=ncount+1


	enddo
      enddo
      dnest=dnest/dble(ncount)
 
      end

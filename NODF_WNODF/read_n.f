      parameter(np=16,nf=6,nrun=100000)
      double precision dp(np,nf), dpi(np,np), dfi(nf,nf)
      double precision dnest, a, dnest_av, dnest_av2, dnest0
      double precision dp_r(np,nf), dpi_r(np,np), dfi_r(nf,nf)
      
      call dran_ini(98981765)
      
      open(1,file='P_matrix_matrix.csv')
      
      do i=1, np
	read(1,*) dp(i,1:nf)
      enddo
      
      
      call nest(dnest,dp,np,nf)
      
      print*,'Nestdeness NODF= ',dnest
      dnest0=dnest
c
c Randomization
c.............................................................
100   continue
      do i=1, np
        do j=1, nf
	  dp_r(i,j)= dp(i,j)
        enddo
      enddo
      
      nc=0
      dnest_av=0.d0
      do irun=1, nrun
      do i=1, 10*np*nf
	i1=i_dran(np)
	j1=i_dran(nf)
	i2=i_dran(np)
	j2=i_dran(nf)

	a=dp_r(i1,j1)
	dp_r(i1,j1)= dp_r(i2,j2)
	dp_r(i2,j2)= a
      enddo

      call nest(dnest,dp_r,np,nf)
      if(dnest.ge.dnest0) nc=nc+1
      
      dnest_av=dnest_av + dnest
      enddo
      print*,'Nestedness rand='  dnest_av/dble(nrun), 'p < ', dble(nc)/dble(nrun)
      
      
      end

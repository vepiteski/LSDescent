all: MATLABcg_default MATLABcg_descent

MATLABcg_default: cg_default.c
	mex cg_default.c

MATLABcg_descent: cg_descent.c
	mex cg_descent.c

clean:
	rm *.mex*

purge:
	rm *.mex*

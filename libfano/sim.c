/* Simulate AWGN channel
 * Copyright 1994 Phil Karn, KA9Q
 */
#include <math.h>

#define	OFFSET	128

double normal_rand(double mean, double std_dev);

/* Turn binary symbols into individual 8-bit channel symbols
 * with specified noise and gain
 */
void
modnoise(
unsigned char *symbols,		/* Input and Output symbols, 8 bits each */
unsigned int nsyms,		/* Symbol count */
double amp,			/* Signal amplitude */
double noise			/* Noise amplitude */
){
	double s;

	while(nsyms-- != 0){
		s = normal_rand(*symbols ? 1.0 : -1.0,noise);
		s *= amp;
		s += OFFSET;
		if(s > 255)
			s = 255;	/* Clip to 8-bit range */
		if(s < 0)
			s = 0;
		*symbols++ = floor(s+0.5);	/* Round to int */
	}
}

#define	MAX_RANDOM	0x7fffffff

/* Generate gaussian random double with specified mean and std_dev */
double
normal_rand(double mean, double std_dev)
{
	double fac,rsq,v1,v2;
	static double gset;
	static int iset;

	if(iset){
		/* Already got one */
		iset = 0;
		return mean + std_dev*gset;
	}
	/* Generate two evenly distributed numbers between -1 and +1
	 * that are inside the unit circle
	 */
	do {
		v1 = 2.0 * (double)random() / MAX_RANDOM - 1;
		v2 = 2.0 * (double)random() / MAX_RANDOM - 1;
		rsq = v1*v1 + v2*v2;
	} while(rsq >= 1.0 || rsq == 0.0);
	fac = sqrt(-2.0*log(rsq)/rsq);
	gset = v1*fac;
	iset++;
	return mean + std_dev*v2*fac;
}

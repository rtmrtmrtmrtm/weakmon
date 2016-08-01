/* Test a rate 1/2 soft decision sequential decoder
 * Copyright 1994 Phil Karn, KA9Q
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <curses.h>

#include "fano.h"

#define	RATE	0.5
#define	OFFSET	128

struct stat {
	unsigned long cycles;
	unsigned long count;
};

main(argc,argv)
int argc;
char *argv[];
{
	double ebn0,mebn0,esn0,mesn0,noise,mnoise,s;
	int mettab[2][256];
	int amp,nbits,i,bit;
	unsigned char *psymbols,*symbols;
	unsigned char *data,*decdata;
	unsigned long metric,cycles,totcycles;
	long t,ntrials;
	long seed;
	extern char *optarg;
	extern int optind;
	unsigned long limit;
	int delta;
	struct stat stats[25];
	unsigned long errors;
	int range;
	FILE *output;
	int quiet;
	int timetest;

	timetest = 0;
	totcycles = 0;
	amp = 100;
	mebn0 = ebn0 = 5.0;
	nbits = 1152;
	ntrials = 10;
	delta = 17;
	limit = 10000;
	quiet = 0;
	time(&seed);
	while((i = getopt(argc,argv,"a:e:n:N:d:l:qs:tm:")) != EOF){
		switch(i){
		case 'a':	/* Signal amplitude in units */
			amp = atoi(optarg);
			break;
		case 'e':	/* Eb/N0 in dB */
			mebn0 = ebn0 = atof(optarg);
			break;
		case 'm':	/* Metric table Eb/N0 in dB */
			mebn0 = atof(optarg);
			break;
		case 'n':	/* Number of data bits */
			nbits = atoi(optarg);
			break;
		case 'N':	/* Number of trials (frames) */
			ntrials = atoi(optarg);
			break;
		case 'd':	/* Decoder threshold adjustment (delta) */
			delta = atoi(optarg);
			break;
		case 'l':	/* Limit on decoder operations/bit */
			limit = atoi(optarg);
			break;
		case 'q':	/* Suppress curses update */
			quiet++;
			break;
		case 's':	/* Seed for random number generator */
			seed = atoi(optarg);
			break;
		case 't':	/* Timetest mode */
			timetest = 1;
			break;
		}
	}
	if(optind >= argc){
		usage();
		exit(1);
	}
	srandom(seed);

	output = fopen(argv[optind],"w+");

	esn0 = ebn0 + 10*log10(RATE);	/* actual Es/N0 in dB */
	mesn0 = mebn0 + 10*log10(RATE);	/* metric table Es/N0 in dB */

	/* Compute noise voltage. The 0.5 factor accounts for BPSK seeing
	 * only half the noise power, and the sqrt() converts power to
	 * voltage.
	 */
	noise = sqrt(0.5/pow(10.,esn0/10.));
	mnoise = sqrt(0.5/pow(10.,mesn0/10.));
	
	data = malloc(nbits/8);
	decdata = malloc(nbits/8);
	psymbols = malloc(nbits*2);
	symbols = malloc(nbits*2);

	/* Generate metrics analytically, with gaussian pdf */	
	gen_met(mettab,amp,mnoise,RATE,4);

	/* Generate data (all 0's) and encode */
	memset(data,0,nbits/8);

#ifndef	notdef
	for(i=0;i<nbits/8 - 5;i++)
		data[i] = 0x55;
#endif

#ifdef	notdef
	data[0] = 0x40;
#endif
        strcpy(data, "hello");
	encode(psymbols,data,nbits/8);
	if(timetest){
		memcpy(symbols,psymbols,2*nbits);
		modnoise(symbols,2*nbits,100.,noise);
		for(t=1;t<=ntrials;t++)
			fano(&metric,&cycles,decdata,symbols,nbits,mettab,delta,limit);
		printf("Ntrials = %ld, cycles = %ld\n",ntrials,cycles);
		exit(0);
	}
	errors = 0;
	memset(stats,0,sizeof(stats));
	i = 0;
	for(range = 1; range < 100000; range *= 10){
		stats[i++].cycles = range;
		stats[i++].cycles = range * 2;
		stats[i++].cycles = range * 4;
		stats[i++].cycles = range * 6;
		stats[i++].cycles = range * 8;
	}
	if(!quiet)
		initscr();
	for(t = 1;t <= ntrials;t++){
		memcpy(symbols,psymbols,2*nbits);
		modnoise(symbols,2*nbits,100.,noise);
		i = fano(&metric,&cycles,decdata,symbols,nbits,mettab,delta,limit);
		totcycles += cycles;
		if(i == 0){
			cycles /= nbits;
			if(memcmp(data,decdata,nbits/8) != 0)
				errors++;
		} else
			cycles = limit + 1;	/* For binning */
		
		for(i=0;i<25;i++){
			if(cycles < stats[i].cycles)
				break;
			stats[i].count++;
		}
		/* Don't write to the file each and every time */
		if((t & 31) == 0 || t == ntrials){
			rewind(output);
			fprintf(output,"Seed %ld Amplitude %d units, Eb/N0 = %lg dB metric table Eb/N0 = %lg dB\n",seed,amp,ebn0,mebn0);
			fprintf(output,"Frame length = %d bits, delta = %d, cycle limit = %ld, #frames = %ld\n",
			  nbits,delta,limit,ntrials);

			fprintf(output,"decoding errors: %ld\n",errors);
			fprintf(output,"Average N: %lf\n",
			 (double)totcycles/stats[0].count/nbits);
			fprintf(output,"  N >=  count fraction\n");
			for(i=0;i<25;i++){
				fprintf(output,"%6d %6ld %.2lg\n",stats[i].cycles,
				 stats[i].count,(double)stats[i].count/t);
				if(stats[i].cycles >= limit || stats[i].count == 0)
					break;
			}

		}
		if(!quiet){

			erase();
			move(0,0);
			printw("Seed %ld Amplitude %d units, Eb/N0 = %lg dB metric table Eb/N0 = %lg dB\n",seed,amp,ebn0,mebn0);
			printw("Frame length = %d bits, delta = %d, cycle limit = %ld, #frames = %ld\n",
			 nbits,delta,limit,ntrials);

			printw("decoding errors: %ld\n",errors);
			printw("Average N: %lf\n",(double)totcycles/stats[0].count/nbits);
			printw("  N >=  count fraction\n");
			for(i=0;i<25;i++){
				printw("%6d %6ld %.2lg\n",stats[i].cycles,
				   stats[i].count,(double)stats[i].count/t);
				if(stats[i].cycles >= limit || stats[i].count == 0)
					break;
			}

			refresh();
		}
	}
	if(!quiet)
		endwin();

	printf("Seed %ld Amplitude %d units, Eb/N0 = %lg dB metric table Eb/N0 = %lg dB\n",seed,amp,ebn0,mebn0);
	printf("Frame length = %d bits, delta = %d, cycle limit = %ld, #frames = %ld\n",
	 nbits,delta,limit,ntrials);
	printf("decoding errors: %ld\n",errors);
	printf("Average N: %lf\n",(double)totcycles/stats[0].count/nbits);
	printf("  N >=  count fraction\n");
	for(i=0;i<25;i++){
		printf("%6d %6ld %.2lg\n",stats[i].cycles,
		   stats[i].count,(double)stats[i].count/ntrials);
		if(stats[i].cycles >= limit || stats[i].count == 0)
			break;
	}
}

usage()
{
	printf("Usage: seqtest [options] output_file\n");
	printf("Option&default 	meaning\n");
	printf("-a 100		signal amplitude in units\n");
	printf("-e 5.0		Signal Eb/N0 in dB (also sets -m)\n");
	printf("-m 5.0		Eb/N0 in dB for metric table calc\n");
	printf("-n 1152		Number of bits per frame\n");
	printf("-N 10		Number of frames to simulate\n");
	printf("-d 17		Decoder threshold (delta)\n");
	printf("-l 10000	Decoder timeout, fwd motions per bit\n");
	printf("-s [cur time]	seed for random number generator\n\n");

	printf("-q		select quiet mode (default off)\n");
	printf("-t		select timetest mode (default off)\n");
}

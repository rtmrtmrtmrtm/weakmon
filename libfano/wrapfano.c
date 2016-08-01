#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <assert.h>

#include "fano.h"

#define RATE 0.5

#define LL 1

#ifdef	LL
/* Layland-Lushbaugh code
 * Nonsystematic, non-quick look-in, dmin=?, dfree=?
 */
#define	POLY1	0xf2d05351
#define	POLY2	0xe4613c47
#endif

/* Convolutional encoder macro. Takes the encoder state, generates
 * a rate 1/2 symbol pair and stores it in 'sym'. The symbol generated from
 * POLY1 goes into the 2-bit of sym, and the symbol generated from POLY2
 * goes into the 1-bit.
 */
#define	ENCODE(sym,encstate){\
	unsigned long _tmp;\
\
	_tmp = (encstate) & POLY1;\
	_tmp ^= _tmp >> 16;\
	(sym) = Partab[(_tmp ^ (_tmp >> 8)) & 0xff] << 1;\
	_tmp = (encstate) & POLY2;\
	_tmp ^= _tmp >> 16;\
	(sym) |= Partab[(_tmp ^ (_tmp >> 8)) & 0xff];\
}

// in_bits should include 31 bits of zero padding at the end.
// there will be 2x as many out_bits as in_bits.
// out_bits[] are 0/1.
// just do the encoding -- the encode() in fano.c really wants bytes.
void
fano_encode(unsigned char in_bits[], int n_in, unsigned char out_bits[])
{
  unsigned long encstate = 0;
  int j;

  j = 0;
  for(int i = 0; i < n_in; i++){
    int sym;
    encstate = (encstate << 1) | in_bits[i];
    ENCODE(sym, encstate);
    out_bits[j++] = sym >> 1;
    out_bits[j++] = sym & 1;
  }
}

// in0[i] is the log of the probability that the i'th symbol is a 0.
// should be 2*n_out in0[] and in1[].
// n_out is number of bits in the decoded message,
// same as n_in passed to fano_encode(),
// presumably including the trailing 31 zeros.
// out_bits are 0/1.
// returns 1 if OK, 0 on error.
int
nfano_decode(int in0[], int in1[], int n_out, unsigned char out_bits[],
             int limit, int out_metric[])
{
  unsigned char decdata[1024];
  unsigned long metric=0;
  unsigned long cycles=0;
  int delta;
  unsigned long maxcycles;
  int ret;

  delta = 17;
  maxcycles = limit; // how hard to work; 5000 to 100000

  ret = nfano(&metric, // output
              &cycles, // output
              decdata, // output, decoded msg, packed into bytes
              in0,
              in1,
              n_out,   // bits expected in the decoded msg
              delta,   // decoder threshold adjustment
              maxcycles);
  if(ret < 0)
    return 0;

  for(int i = 0; i < n_out; i++)
    out_bits[i] = (decdata[i/8] >> (7 - (i % 8))) & 1;

  out_metric[0] = metric;

  return 1;
}

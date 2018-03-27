//
// LDPC decoder for FT8.
//
// given a 174-bit codeword as an array of log-likelihood of zero,
// return a 174-bit corrected codeword, or zero-length array.
// last 87 bits are the (systematic) plain-text.
// this is an implementation of the sum-product algorithm
// from Sarah Johnson's Iterative Error Correction book.
// codeword[i] = log ( P(x=0) / P(x=1) )
//
// cc -O3 libldpc.c -shared -fPIC -o libldpc.so
//

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "arrays.h"

int ldpc_check(int codeword[]);

// thank you Douglas Bagnall
// https://math.stackexchange.com/a/446411
float fast_tanh(float x){
  if(x < -4.97){
    return -1.0;
  }
  if(x > 4.97){
    return 1.0;
  }
  float x2 = x * x;
  float a = x * (135135.0f + x2 * (17325.0f + x2 * (378.0f + x2)));
  float b = 135135.0f + x2 * (62370.0f + x2 * (3150.0f + x2 * 28.0f));
  return a / b;
}

// codeword is 174 log-likelihoods.
// plain is a return value, 174 ints, to be 0 or 1.
// iters is how hard to try.
// ok == 87 means success.
void
ldpc_decode(double codeword[], int iters, int plain[], int *ok)
{
  double m[87][174];
  double e[87][174];
  int best_score = -1;
  int best_cw[174];
  
  for(int i = 0; i < 174; i++)
    for(int j = 0; j < 87; j++)
      m[j][i] = codeword[i];

  for(int i = 0; i < 174; i++)
    for(int j = 0; j < 87; j++)
      e[j][i] = 0.0;

  for(int iter = 0; iter < iters; iter++){
    for(int j = 0; j < 87; j++){
      for(int ii1 = 0; ii1 < 7; ii1++){
        int i1 = Nm[j][ii1] - 1;
        if(i1 < 0)
          continue;
        double a = 1.0;
        for(int ii2 = 0; ii2 < 7; ii2++){
          int i2 = Nm[j][ii2] - 1;
          if(i2 >= 0 && i2 != i1){
            a *= fast_tanh(m[j][i2] / 2.0);
          }
        }
        e[j][i1] = log((1 + a) / (1 - a));
      }
    }
          
    int cw[174];
    for(int i = 0; i < 174; i++){
      double l = codeword[i];
      for(int j = 0; j < 3; j++)
        l += e[Mn[i][j]-1][i];
      cw[i] = (l <= 0.0);
    }
    int score = ldpc_check(cw);
    if(score == 87){
#if 0
      int cw1[174];
      for(int i = 0; i < 174; i++)
        cw1[i] = cw[colorder[i]];
      for(int i = 0; i < 87; i++)
        plain[i] = cw1[174-87+i];
#else
      for(int i = 0; i < 174; i++)
        plain[i] = cw[colorder[i]];
#endif
      *ok = 87;
      return;
    }

    if(score > best_score){
      for(int i = 0; i < 174; i++)
        best_cw[i] = cw[i];
      best_score = score;
    }

    for(int i = 0; i < 174; i++){
      for(int ji1 = 0; ji1 < 3; ji1++){
        int j1 = Mn[i][ji1] - 1;
        double l = codeword[i];
        for(int ji2 = 0; ji2 < 3; ji2++){
          if(ji1 != ji2){
            int j2 = Mn[i][ji2] - 1;
            l += e[j2][i];
          }
        }
        m[j1][i] = l;
      }
    }
  }

  // decode didn't work, return something anyway.
#if 0
  int cw1[174];
  for(int i = 0; i < 174; i++)
    cw1[i] = best_cw[colorder[i]];
  for(int i = 0; i < 87; i++)
    plain[i] = cw1[174-87+i];
#else
  for(int i = 0; i < 174; i++)
    plain[i] = best_cw[colorder[i]];
#endif

  *ok = best_score;
}

//
// does a 174-bit codeword pass the FT8's LDPC parity checks?
// returns the number of parity checks that passed.
// 87 means total success.
//
int
ldpc_check(int codeword[])
{
  int score = 0;
  
  // Nm[87][7]
  for(int j = 0; j < 87; j++){
    int x = 0;
    for(int ii1 = 0; ii1 < 7; ii1++){
      int i1 = Nm[j][ii1] - 1;
      if(i1 >= 0){
        x ^= codeword[i1];
      }
    }
    if(x == 0)
      score++;
  }
  return score;
}

void
ft8_crc(int msg1[], int msglen, int out[12])
{
  // the FT8 polynomial for 12-bit CRC.
  int div[] = { 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0 };

  // append 12 zeros.
  int *msg = (int *)malloc(sizeof(int) * (msglen + 12));
  for(int i = 0; i < msglen + 12; i++){
    if(i < msglen){
      msg[i] = msg1[i];
    } else {
      msg[i] = 0;
    }
  }

  for(int i = 0; i < msglen; i++){
    if(msg[i]){
      for(int j = 0; j < 13; j++){
        msg[i+j] = (msg[i+j] + div[j]) % 2;
      }
    }
  }

  for(int i = 0; i < 12; i++){
    out[i] = msg[msglen+i];
  }

  free(msg);
}

// rows is 87.
// m[174][174]. 
// m's right half should start out as zeros.
// m's upper-right quarter will be the desired inverse.
void
gauss_jordan(int rows, int m[2*rows][2*rows], int which[rows], int *ok)
{
  *ok = 0;

  for(int row = 0; row < rows; row++){
    if(m[row][row] != 1){
      for(int row1 = row+1; row1 < 2*rows; row1++){
        if(m[row1][row] == 1){
          // swap m[row] and m[row1]
          for(int col = 0; col < 2*rows; col++){
            int tmp = m[row][col];
            m[row][col] = m[row1][col];
            m[row1][col] = tmp;
          }
          int tmp = which[row];
          which[row] = which[row1];
          which[row1] = tmp;
          break;
        }
      }
    }
    if(m[row][row] != 1){
      // could not invert
      *ok = 0;
      return;
    }
    // lazy creation of identity matrix in the upper-right quarter
    m[row][rows+row] = (m[row][rows+row] + 1) % 2;
    // now eliminate
    for(int row1 = 0; row1 < 2*rows; row1++){
      if(row1 == row)
        continue;
      if(m[row1][row] != 0){
        for(int col = 0; col < 2*rows; col++){
          m[row1][col] = (m[row1][col] + m[row][col]) % 2;
        }
      }
    }
  }

  *ok = 1;
}

//  # given a 174-bit codeword as an array of log-likelihood of zero,
//  # return a 87-bit plain text, or zero-length array.
//  # this is an implementation of the sum-product algorithm
//  # from Sarah Johnson's Iterative Error Correction book.
//  # codeword[i] = log ( P(x=0) / P(x=1) )
//  def ldpc_decode(self, codeword):
//      # 174 codeword bits
//      # 87 parity checks
//
//      # Mji
//      # each codeword bit i tells each parity check j
//      # what the bit's log-likelihood of being 0 is
//      # based on information *other* than from that
//      # parity check.
//      m = numpy.zeros((87, 174))
//
//      # Eji
//      # each check j tells each codeword bit i the
//      # log likelihood of the bit being zero based
//      # on the *other* bits in that check.
//      e = numpy.zeros((87, 174))
//
//      for i in range(0, 174):
//          for j in range(0, 87):
//              m[j][i] = codeword[i]
//
//      for iter in range(0, 50):
//          # messages from checks to bits.
//          # for each parity check
//          for j in range(0, 87):
//              # for each bit mentioned in this parity check
//              for i in Nm[j]:
//                  if i <= 0:
//                      continue
//                  a = 1
//                  # for each other bit mentioned in this parity check
//                  for ii in Nm[j]:
//                      if ii != i:
//                          a *= math.tanh(m[j][ii-1] / 2.0)
//                  e[j][i-1] = math.log((1 + a) / (1 - a))
//
//          # decide if we are done -- compute the corrected codeword,
//          # see if the parity check succeeds.
//          cw = numpy.zeros(174, dtype=numpy.int32)
//          for i in range(0, 174):
//              # sum the log likelihoods for codeword bit i being 0.
//              l = codeword[i]
//              for j in Mn[i]:
//                  l += e[j-1][i]
//              if l > 0:
//                  cw[i] = 0
//              else:
//                  cw[i] = 1
//          if self.ldpc_check(cw):
//              # success!
//              # it's a systematic code, though the plain-text bits are scattered.
//              # collect them.
//              decoded = cw[colorder]
//              decoded = decoded[-87:]
//              return decoded
//
//          # messages from bits to checks.
//          for i in range(0, 174):
//              for j in Mn[i]:
//                  l = codeword[i]
//                  for jj in Mn[i]:
//                      if jj != j:
//                          l += e[jj-1][i]
//                  m[j-1][i] = l
//
//      # could not decode.
//      return numpy.array([])

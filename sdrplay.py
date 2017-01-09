#
# control an SDRPlay RSP, and read I/Q samples.
#
# uses the SDRPlay API library (libmirsdrapi-rsp.so).
#
# Robert Morris, AB1HL
#

import ctypes
import time
import thread
import sys
import numpy

#
# if already connected, return existing SDRplay,
# otherwise a new one.
#
the_sdrplay = None
mu = thread.allocate_lock()
def open(dev):
    global the_sdrplay, mu
    mu.acquire()
    if the_sdrplay == None:
        the_sdrplay = SDRplay()
    sdr = the_sdrplay
    mu.release()
    return sdr

class SDRplay:
    def __init__(self):
        # SDRplay config.
        self.samplerate = 2048000 # what to tell the SDRplay.
        self.decimate = 8 # tell SDRplay to give us every 8th sample.
        self.use_callback = True # use new API callback rather than ReadPacket

        # callback appends incoming buffers here.
        # each element is [ i[], q[] ].
        self.cb_bufs = [ ]
        self.cb_seq = None # next sample num expected by callback
        self.cb_time = time.time() # UNIX time of end of cb_bufs
        self.cb_bufs_mu = thread.allocate_lock()

        # on mac, this must exist: /usr/local/lib/libusb-1.0.0.dylib
        # on linux, setenv LD_LIBRARY_PATH /usr/local/lib
        # on linux, may need to be run as root to get at USB device.

        # try a few different names for the library
        names = [
            "/usr/local/lib/libmirsdrapi-rsp.so.1.95",
            "libmirsdrapi-rsp.so.1.95",
            "libmirsdrapi-rsp.so",
            "./libmirsdrapi-rsp.so.1.95",
            ]
        self.lib = None
        for name in names:
            try:
                self.lib = ctypes.cdll.LoadLibrary(name)
                print "sdrplay: loaded API from %s" % (name)
                break
            except:
                pass
        if self.lib == None:
            sys.stderr.write("sdrplay: could not load API library libmisdrapi-rsp.so\n")
            sys.exit(1)
        
        vers = ctypes.c_float(0.0)
        self.lib.mir_sdr_ApiVersion(ctypes.byref(vers))
        if vers.value < 1.95-0.00001 or vers.value > 1.95+0.00001:
            sys.stderr.write("sdrplay.py: warning: needs API version 1.95, got %f" % (vers.value))

        #self.lib.mir_sdr_DebugEnable(1)

        sps = ctypes.c_int(0)

        # type of the callback function
        t1 = ctypes.CFUNCTYPE(ctypes.c_int,
                              ctypes.POINTER(ctypes.c_int16), # xi
                              ctypes.POINTER(ctypes.c_int16), # xq
                              ctypes.c_int, # firstSampleNum
                              ctypes.c_int, # grChanged
                              ctypes.c_int, # rfChanged
                              ctypes.c_int, # fsChanged
                              ctypes.c_uint, # numSamples
                              ctypes.c_uint, # reset
                              ctypes.c_void_p) # cbContext
        self.cb1 = t1(self.callback)

        t2 = ctypes.CFUNCTYPE(ctypes.c_int)
        self.cb2 = t2(self.callbackGC)

        if self.use_callback:
            # new streaming/callback API
            self.lib.mir_sdr_DecimateControl(1, self.decimate, 0)
            self.lib.mir_sdr_AgcControl(1, -30, 0, 0, 0, 0, 0);
            newGr = ctypes.c_int(40)
            sysGr = ctypes.c_int(40)
            err = self.lib.mir_sdr_StreamInit(ctypes.byref(newGr),
                                              ctypes.c_double(self.samplerate / 1000000.0),  # sample rate, millions
                                              ctypes.c_double(1000.0),  # center frequency, MHz
                                              200,    # mir_sdr_BW_0_200
                                              0,      # mir_sdr_IF_Zero
                                              0,      # LNAEnable
                                              ctypes.byref(sysGr),
                                              1,      # useGrAltMode
                                              ctypes.byref(sps),
                                              self.cb1,
                                              self.cb2,
                                              0)
            if err != 0:
                sys.stderr.write("sdrplay: mir_sdr_StreamInit failed: %d\n" % (err))
                sys.exit(1)
        else:
            # older ReadPacket API
            err = self.lib.mir_sdr_Init(40, # gRdB
                                        ctypes.c_double(self.samplerate / 1000000.0),  # sample rate, millions
                                        ctypes.c_double(14.076),  # center frequency, MHz
                                        200,    # mir_sdr_BW_0_200
                                        0,      # mir_sdr_IF_Zero
                                        ctypes.byref(sps)) # samplesPerPacket
            if err != 0:
                sys.stderr.write("sdrplay: mir_sdr_Init failed: %d\n" % (err))
                sys.exit(1)
                                        
            
        self.samplesPerPacket = sps.value
        self.expect = None

        # what does this do?
        self.lib.mir_sdr_SetDcMode(4, 0)
        self.lib.mir_sdr_SetDcTrackTime(63)

    def setfreq(self, hz):
        # this doesn't work if you're changing bands.
        # err = self.lib.mir_sdr_SetRf(ctypes.c_double(float(hz)), 1, 0)
        # if err != 0:
        #     sys.stderr.write("sdrplay: SetRf(%d) failed: %d\n" % (hz, err))

        newGr = ctypes.c_int(40)
        sysGr = ctypes.c_int(40)
        sps = ctypes.c_int(0)
        err = self.lib.mir_sdr_Reinit(ctypes.byref(newGr), # gRdB
                                      ctypes.c_double(0.0), # fsMHz
                                      ctypes.c_double(hz / 1000000.0), # rfMHz
                                      0, # bwType,
                                      0, # ifType
                                      0, # LoMode
                                      0, # LNAEnable
                                      ctypes.byref(sysGr), # gRdBsystem,
                                      1, # useGrAltMode
                                      ctypes.byref(sps),
                                      0x04) # mir_sdr_CHANGE_RF_FREQ
        if err != 0:
            sys.stderr.write("sdrplay: mir_sdr_Reinig(rfMHz=%f) failed: %d\n" % (hz/1000000.0, err))
                                      

    # for weakaudio. probably 256000.
    def getrate(self):
        if self.use_callback:
            return self.samplerate / float(self.decimate)
        else:
            return self.samplerate

    def callback(self, xi, xq, firstSampleNum, grChanged,
                 rfChanged, fsChanged, numSamples, reset,
                 cbContext):

        # theory: firstSampleNum is 16 bits, and upper bits are junk.
        firstSampleNum &= 0xffff

        # ii = [ xi[i] for i in range(0, numSamples) ]
        # qq = [ xq[i] for i in range(0, numSamples) ]

        istring = ctypes.string_at(ctypes.addressof(xi.contents), numSamples * 2)
        ii = numpy.fromstring(istring, dtype=numpy.int16)

        qstring = ctypes.string_at(ctypes.addressof(xq.contents), numSamples * 2)
        qq = numpy.fromstring(qstring, dtype=numpy.int16)

        self.cb_bufs_mu.acquire()

        if reset != 0:
            self.cb_seq = None
            self.cb_time = time.time()
        
        if self.cb_seq != None and self.cb_seq != firstSampleNum:
            print "SDRplay callback missed %d samples; %d %d %d" % (firstSampleNum - self.cb_seq,
              self.cb_seq, firstSampleNum, numSamples)

        self.cb_bufs.append([ ii, qq ])
        self.cb_seq = (firstSampleNum + numSamples) & 0xffff
        self.cb_time += numSamples / float(self.getrate())

        self.cb_bufs_mu.release()

        return 0

    def callbackGC(self):
        return 0

    # internal, for old non-callback API.
    # returns I/Q as a numpy complex array.
    def rawread(self):
        aty = ctypes.c_int16 * self.samplesPerPacket
        xi = aty()
        xq = aty()
        firstSampleNum = ctypes.c_uint(0)
        grChanged = ctypes.c_uint(0)
        rfChanged = ctypes.c_uint(0)
        fsChanged = ctypes.c_uint(0)
        err = self.lib.mir_sdr_ReadPacket(ctypes.byref(xi),
                                          ctypes.byref(xq),
                                          ctypes.byref(firstSampleNum),
                                          ctypes.byref(grChanged),
                                          ctypes.byref(rfChanged),
                                          ctypes.byref(fsChanged))
        if err != 0:
            sys.stderr.write("sdrplay: mir_sdrReadPacket failed: %d\n" % (err))
            sys.exit(1)

        # I don't know if these are needed.
        if grChanged.value != 0:
            self.lib.mir_sdr_ResetUpdateFlags(1, 0, 0)
        if rfChanged.value != 0:
            self.lib.mir_sdr_ResetUpdateFlags(0, 1, 0)
        if fsChanged.value != 0:
            self.lib.mir_sdr_ResetUpdateFlags(0, 0, 1)

        # ii = numpy.fromstring(xi, dtype=numpy.int16)
        # qq = numpy.fromstring(xq, dtype=numpy.int16)

        ii = [ xi[i] for i in range(0, self.samplesPerPacket) ]
        ii = numpy.array(ii).astype(numpy.float64)

        qq = [ xq[i] for i in range(0, self.samplesPerPacket) ]
        qq = numpy.array(qq).astype(numpy.float64)

        iq = ii + 1j*qq

        # theory: firstSampleNum is only 16 bits, upper bits are junk.
        num = firstSampleNum.value
        num = num & 0xffff
        if self.expect != None and num != self.expect:
            print "%d vs %d -- gap %d" % (self.expect, num, num - self.expect)

        self.expect = (num + self.samplesPerPacket) & 0xffff

        return iq

    # blocking read.
    # returns [ samples, end_time ]
    def readiq(self):
        if self.use_callback:
            while True:
                self.cb_bufs_mu.acquire()
                bufbuf = self.cb_bufs
                end_time = self.cb_time
                self.cb_bufs = [ ]
                self.cb_bufs_mu.release()
            
                if len(bufbuf) > 0:
                    break
                time.sleep(0.1)

                
            bufs = [ ]
            for e in bufbuf:
                ii = numpy.array(e[0]).astype(numpy.float64)
                qq = numpy.array(e[1]).astype(numpy.float64)
                iq = ii + 1j*qq
                bufs.append(iq)

            buf = numpy.concatenate(bufs)
        else:
            buf = self.rawread()

        buf = buf / 10.0 # get rid of spurious clip warnings

        return [ buf, end_time ]

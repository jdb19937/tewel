PICS = /spin/dan/dne.512
VID = /spin/dan/trolls.dat
DIM = 64x64
LIMIT = 4096
REPS = 1000000
TEWEL = tewel
TGT = gen.ctx dis.ctx opt.txt sty.dat src.dat

.PHONY: all
all: $(TGT)

.PHONY: bootstrap
bootstrap: all
	$(TEWEL) learnauto . reps=32

.PHONY: learn
learn: all
	$(TEWEL) learnhans . reps=4096

.PHONY: clean
clean:
	rm -f $(TGT)


gen.ctx: gen.txt
	$(TEWEL) make gen.ctx spec=gen.txt clobber=1

dis.ctx: dis.txt
	$(TEWEL) make dis.ctx spec=dis.txt clobber=1

opt.txt:
	echo src=src.dat sty=sty.dat dim=$(DIM) > $@


sty.dat:
	TEWEL_VIDREADER_SCALE=320:-1 \
	$(TEWEL) synth //identity.proj \
		verbose=1 crop=random \
		dim=$(DIM) src=$(VID) out=$@

src.dat:
	$(TEWEL) synth //identity.proj \
		verbose=1 lowmem=1 srctrav=rand crop=random \
		dim=$(DIM) limit=$(LIMIT) src=$(PICS) out=$@

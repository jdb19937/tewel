PICS = /spin/dan/dne.512
FILTER = //degrade4x.sh
DIM = 64x64
LIMIT = 4096
REPS = 1000000
TEWEL = tewel
TGT = gen.ctx dis.ctx opt.txt tgt/pics.dat src/pics.dat

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
	rm -rf src tgt


gen.ctx: gen.txt
	$(TEWEL) make gen.ctx spec=gen.txt clobber=1

dis.ctx: dis.txt
	$(TEWEL) make dis.ctx spec=dis.txt clobber=1

opt.txt:
	echo dim=$(DIM) > $@


tgt/pics.dat:
	mkdir tgt
	$(TEWEL) synth //identity.proj \
		randseed=1 verbose=1 lowmem=1 srctrav=rand crop=random \
		dim=$(DIM) limit=$(LIMIT) src=$(PICS) out=$@

src/pics.dat:
	mkdir src
	TEWEL_PICREADER_FILTER=$(FILTER) \
	$(TEWEL) synth //identity.proj \
		randseed=1 verbose=1 lowmem=1 srctrav=rand crop=random \
		dim=$(DIM) limit=$(LIMIT) src=$(PICS) out=$@

PICS = /spin/dan/dne.512
FILTER = //degrade4x.sh
DIM = 64x64
LIMIT = 65536
REPS = 1000000
TEWEL = tewel
TGT = gen.ctx dis.ctx opt.txt tgt.dat src.dat

.PHONY: all
all: $(TGT)

.PHONY: learn
learn: all
	$(TEWEL) learnfunc . reps=128
	$(TEWEL) learnhans . reps=65536

.PHONY: clean
clean:
	rm -f $(TGT)


gen.ctx: gen.txt
	$(TEWEL) make gen.ctx spec=gen.txt clobber=1

dis.ctx: dis.txt
	$(TEWEL) make dis.ctx spec=dis.txt clobber=1

opt.txt:
	echo src=src.dat tgt=tgt.dat dim=$(DIM) > $@


tgt.dat:
	$(TEWEL) synth //identity.proj \
		randseed=1 verbose=1 lowmem=1 srctrav=rand crop=random \
		dim=$(DIM) limit=$(LIMIT) src=$(PICS) out=$@

src.dat:
	TEWEL_PICREADER_FILTER=$(FILTER) \
	$(TEWEL) synth //identity.proj \
		randseed=1 verbose=1 lowmem=1 srctrav=rand crop=random \
		dim=$(DIM) limit=$(LIMIT) src=$(PICS) out=$@

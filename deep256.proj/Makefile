DIM = 256x256
LIMIT = 65536
REPS = 1000000
TEWEL = tewel
TGT = gen.ctx dis.ctx enc.ctx tgt.dat src.dat

.PHONY: all
all: $(TGT)

.PHONY: learn
learn: all
	$(TEWEL) learnfunc . src=src.dat tgt=tgt.dat dim=$(DIM)x7 addgeo=1 reps=65536

.PHONY: normalize
normalize:
	$(TEWEL) normalize . src=tgt.dat srcdim=$(DIM)x3 addgeo=1 verbose=1

.PHONY: clean
clean:
	rm -f $(TGT)


gen.ctx: gen.txt
	$(TEWEL) make gen.ctx spec=gen.txt clobber=1

dis.ctx: dis.txt
	$(TEWEL) make dis.ctx spec=dis.txt clobber=1

enc.ctx: enc.txt
	$(TEWEL) make enc.ctx spec=enc.txt clobber=1

nova2.256.dat: /spin/dan/nova2.aligned
	$(TEWEL) synth //identity.proj \
		randseed=1 verbose=1 lowmem=1 srctrav=scan crop=center \
		dim=$(DIM) limit=$(LIMIT) src=$< out=$@
cand.256.dat: /spin/dan/cand.aligned
	$(TEWEL) synth //identity.proj \
		randseed=1 verbose=1 lowmem=1 srctrav=scan crop=center \
		dim=$(DIM) limit=$(LIMIT) src=$< out=$@
shampane.256.dat: /spin/dan/shampane.aligned
	$(TEWEL) synth //identity.proj \
		randseed=1 verbose=1 lowmem=1 srctrav=scan crop=center \
		dim=$(DIM) limit=$(LIMIT) src=$< out=$@
lfw.256.dat: /spin/dan/lfw.aligned
	$(TEWEL) synth //identity.proj \
		randseed=1 verbose=1 lowmem=1 srctrav=scan crop=center \
		dim=$(DIM) limit=$(LIMIT) src=$< out=$@
miniceleb.256.dat: /spin/dan/miniceleb.aligned
	$(TEWEL) synth //identity.proj \
		randseed=1 verbose=1 lowmem=1 srctrav=scan crop=center \
		dim=$(DIM) limit=$(LIMIT) src=$< out=$@
minidne.256.dat: /spin/dan/minidne.aligned
	$(TEWEL) synth //identity.proj \
		randseed=1 verbose=1 lowmem=1 srctrav=scan crop=center \
		dim=$(DIM) limit=$(LIMIT) src=$< out=$@

tgt.dat: nova2.256.dat cand.256.dat shampane.256.dat lfw.256.dat miniceleb.256.dat minidne.256.dat
	cat $^ > $@

src.dat: tgt.dat
	ln -s $< $@

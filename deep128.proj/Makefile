PICS = /spin/dan/nova.aligned
DIM = 128x128
LIMIT = 65536
REPS = 1000000
TEWEL = tewel
TGT = gen.ctx enc.ctx tgt.dat src.dat

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

enc.ctx: enc.txt
	$(TEWEL) make enc.ctx spec=enc.txt clobber=1

nova2.128.dat: /spin/dan/nova2.aligned
	TEWEL_PICREADER_FILTER=//shrink2x.sh $(TEWEL) synth //identity.proj \
		randseed=1 verbose=1 lowmem=1 srctrav=scan crop=center \
		dim=$(DIM) limit=$(LIMIT) src=$< out=$@
cand.128.dat: /spin/dan/cand.aligned
	TEWEL_PICREADER_FILTER=//shrink2x.sh $(TEWEL) synth //identity.proj \
		randseed=1 verbose=1 lowmem=1 srctrav=scan crop=center \
		dim=$(DIM) limit=$(LIMIT) src=$< out=$@
shampane.128.dat: /spin/dan/shampane.aligned
	TEWEL_PICREADER_FILTER=//shrink2x.sh $(TEWEL) synth //identity.proj \
		randseed=1 verbose=1 lowmem=1 srctrav=scan crop=center \
		dim=$(DIM) limit=$(LIMIT) src=$< out=$@
lfw.128.dat: /spin/dan/lfw.aligned
	TEWEL_PICREADER_FILTER=//shrink2x.sh $(TEWEL) synth //identity.proj \
		randseed=1 verbose=1 lowmem=1 srctrav=scan crop=center \
		dim=$(DIM) limit=$(LIMIT) src=$< out=$@
miniceleb.128.dat: /spin/dan/miniceleb.aligned
	TEWEL_PICREADER_FILTER=//shrink2x.sh $(TEWEL) synth //identity.proj \
		randseed=1 verbose=1 lowmem=1 srctrav=scan crop=center \
		dim=$(DIM) limit=$(LIMIT) src=$< out=$@
minidne.128.dat: /spin/dan/minidne.aligned
	TEWEL_PICREADER_FILTER=//shrink2x.sh $(TEWEL) synth //identity.proj \
		randseed=1 verbose=1 lowmem=1 srctrav=scan crop=center \
		dim=$(DIM) limit=$(LIMIT) src=$< out=$@

tgt.dat: nova2.128.dat cand.128.dat shampane.128.dat lfw.128.dat miniceleb.128.dat minidne.128.dat
	cat $^ > $@

src.dat: tgt.dat
	ln -s $< $@

%.rng: %.128.dat
	$(TEWEL) synth enc.ctx addgeo=1 dim=$(DIM)x7 src=$< out=$@ verbose=1

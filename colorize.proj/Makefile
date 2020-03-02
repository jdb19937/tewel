VID = /spin/dan/trolls.avi
DIM = 64x64
LIMIT = 65536
REPS = 1000000
TEWEL = tewel
TGT = gen.ctx dis.ctx tgt.dat src.dat

.PHONY: all
all: $(TGT)

.PHONY: learn
learn: all
	$(TEWEL) learnfunc . src=src.dat tgt=tgt.dat dim=$(DIM) reps=256
	$(TEWEL) learnhans . src=src.dat tgt=tgt.dat dim=$(DIM) reps=65536 noise=2.0

.PHONY: clean
clean:
	rm -f $(TGT)


gen.ctx: gen.txt
	$(TEWEL) make gen.ctx spec=gen.txt clobber=1

dis.ctx: dis.txt
	$(TEWEL) make dis.ctx spec=dis.txt clobber=1

tgt.dat:
	TEWEL_VIDREADER_RATE=16 \
	TEWEL_VIDREADER_VF="scale=320:-1" \
	$(TEWEL) synth //identity.proj \
		randseed=1 verbose=1 crop=random \
		dim=$(DIM) src=$(VID) out=$@
src.dat:
	TEWEL_VIDREADER_RATE=16 \
	TEWEL_VIDREADER_VF="scale=320:-1,hue=s=0" \
	$(TEWEL) synth //identity.proj \
		randseed=1 verbose=1 crop=random \
		dim=$(DIM) src=$(VID) out=$@

sanfran_1920_color.mp4: /spin/dan/sanfran_1920.mp4
	TEWEL_VIDREADER_VF="scale=320:-1,hue=s=0" \
	tewel synth . src=$< out=$@ verbose=1

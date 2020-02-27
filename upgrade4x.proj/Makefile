FILTER = //degrade4x.sh
PICS = /spin/dan/dne.512
DIM = 64x64
TGT = gen.ctx dis.ctx opt.txt tgt/pics.dat src/pics.dat

all: $(TGT)

clean:
	rm -f $(TGT)
	rm -rf src tgt

gen.ctx: gen.txt
	tewel make gen.ctx spec=gen.txt clobber=1

dis.ctx: dis.txt
	tewel make dis.ctx spec=dis.txt clobber=1

opt.txt:
	echo dim=$(DIM) > $@


tgt/pics.dat:
	mkdir tgt
	tewel synth //identity.proj \
		randseed=1 verbose=1 lowmem=1 \
		dim=$(DIM) srctrav=rand crop=random limit=65536 \
		src=$(PICS) out=$@

src/pics.dat:
	mkdir src
	TEWEL_PICREADER_FILTER=$(FILTER) \
	tewel synth //identity.proj \
		randseed=1 verbose=1 lowmem=1 \
		dim=$(DIM) srctrav=rand crop=random limit=65536 \
		src=$(PICS) out=$@

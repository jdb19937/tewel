.PHONY: all learn1 learn2 learn3 learn4 clean normalize rvg clean1 clean2 clean3 clean4 show1 show2 show3 server




all: learn1 learn2 learn3 learn4




learn1: learn1.done

learn1.done: enc1.ctx gen1.ctx src1.rgb tgt1.rgb
	tewel learnfunc enc1.ctx gen1.ctx src=src1.rgb tgt=tgt1.rgb dim=128x128 mul=1 stoploss=0.05 verbose=1
	tewel learnfunc enc1.ctx gen1.ctx src=src1.rgb tgt=tgt1.rgb dim=128x128 mul=0.25 stoploss=0.045 verbose=1
	touch $@

tgt1.rgb: iden.ctx
	tewel synth iden.ctx verbose=1 dim=128x128 out=$@ src=/spin/dan/lq.128

src1.rgb: tgt1.rgb
	cp -f $< $@







SRC2DIM = 32x32x3
TGT2DIM = 64x64x3
SRC2 = src2.$(SRC2DIM).u8
TGT2 = tgt2.$(TGT2DIM).u8

learn2: learn2.done

learn2.done: $(SRC2) $(TGT2) gen2.ctx dis2.ctx cnd2.ctx
	tewel learnfunc gen2.ctx src=$(SRC2) tgt=$(TGT2) srcdim=$(SRC2DIM) dim=$(SRC2DIM) tgtdim=$(TGT2DIM) stoprounds=32768
	tewel learnhans gen2.ctx src=$(SRC2) tgt=$(TGT2) dis=dis2.ctx cnd=cnd2.ctx dim=$(SRC2DIM) srcdim=$(SRC2DIM) tgtdim=$(TGT2DIM) stoprounds=4000000
	tewel learnhans gen2.ctx src=$(SRC2) tgt=$(TGT2) dis=dis2.ctx cnd=cnd2.ctx dim=$(SRC2DIM) srcdim=$(SRC2DIM) tgtdim=$(TGT2DIM) stoprounds=4032768 mul=0.25
	touch $@

src2.tmp.u8: enc1.ctx gen1.ctx dns1.ctx mean.ctx iden.ctx learn1.done
	tewel synth dns1.ctx mean.ctx enc1.ctx gen1.ctx \
		dim=256x256x3 verbose=1 src=/spin/dan/minidne.aligned out=$@

$(SRC2): src2.tmp.u8 iden.ctx
	tewel synth iden.ctx \
		randseed=1 verbose=1 lowmem=1 srctrav=rand crop=random trim=0.125 \
		srcdim=128x128x3 dim=32x32x3 limit=65536 src=src2.tmp.u8 out=$@

tgt2.tmp.u8: dns1.ctx
	tewel synth dns1.ctx \
		dim=256x256x3 verbose=1 src=/spin/dan/minidne.aligned out=$@

$(TGT2): tgt2.tmp.u8 ups1.ctx
	tewel synth ups1.ctx \
		randseed=1 verbose=1 lowmem=1 srctrav=rand crop=random trim=0.125 \
		srcdim=128x128x12 dim=32x32x12 limit=65536 src=tgt2.tmp.u8 out=$@




SRC3DIM = 32x32x3
TGT3DIM = 64x64x3
SRC3 = src3.$(SRC3DIM).u8
TGT3 = tgt3.$(TGT3DIM).u8


learn3: learn3.done

learn3.done: $(SRC3) $(TGT3) gen3.ctx dis3.ctx cnd3.ctx
	tewel learnfunc gen3.ctx src=$(SRC3) tgt=$(TGT3) srcdim=$(SRC3DIM) dim=$(SRC3DIM) tgtdim=$(TGT3DIM) stoprounds=32768
	tewel learnhans gen3.ctx src=$(SRC3) tgt=$(TGT3) dis=dis3.ctx cnd=cnd3.ctx dim=$(SRC3DIM) srcdim=$(SRC3DIM) tgtdim=$(TGT3DIM) stoprounds=4000000
	tewel learnhans gen3.ctx src=$(SRC3) tgt=$(TGT3) dis=dis3.ctx cnd=cnd3.ctx dim=$(SRC3DIM) srcdim=$(SRC3DIM) tgtdim=$(TGT3DIM) stoprounds=4032768 mul=0.25
	touch $@

src3.tmp.u8: dns1.ctx mean.ctx
	tewel synth dns1.ctx mean.ctx \
		dim=512x512x3 verbose=1 src=/spin/dan/dne.512 out=$@

$(SRC3): src3.tmp.u8 iden.ctx
	tewel synth iden.ctx \
		randseed=1 verbose=1 lowmem=1 srctrav=rand crop=random \
		srcdim=256x256x3 dim=32x32x3 limit=65536 src=src3.tmp.u8 out=$@

tgt3.tmp.u8: dns1.ctx
	tewel synth dns1.ctx \
		dim=512x512x3 verbose=1 src=/spin/dan/dne.512 out=$@

$(TGT3): tgt3.tmp.u8 ups1.ctx
	tewel synth ups1.ctx \
		randseed=1 verbose=1 lowmem=1 srctrav=rand crop=random \
		srcdim=256x256x12 dim=32x32x12 limit=65536 src=tgt3.tmp.u8 out=$@



SRC4DIM = 32x32x3
STY4DIM = 64x64x3
SRC4 = src4.$(SRC4DIM).u8
STY4 = sty4.$(STY4DIM).u8


learn4: learn4.done
learn4.done: gen4.ctx dis4.ctx $(SRC4) $(STY4) pregen4.ctx
	tewel learnfunc pregen4.ctx gen4.ctx src=$(SRC4) tgt=$(SRC4) srcdim=$(SRC4DIM) dim=$(SRC4DIM) tgtdim=$(SRC4DIM) stoprounds=32768
	tewel learnstyl gen4.ctx src=$(SRC4) sty=$(STY4) dis=dis4.ctx dim=$(SRC4DIM) srcdim=$(SRC4DIM) stydim=$(STY4DIM) stoprounds=4000000
	tewel learnstyl gen4.ctx src=$(SRC4) sty=$(STY4) dis=dis4.ctx dim=$(SRC4DIM) srcdim=$(SRC4DIM) stydim=$(STY4DIM) stoprounds=4032768 mul=0.25
	touch $@

$(SRC4): iden.ctx
	tewel synth iden.ctx \
		randseed=1 verbose=1 lowmem=1 srctrav=rand crop=random \
		dim=32x32x3 limit=65536 src=/spin/dan/dne.512 out=$@

$(STY4): iden.ctx
	tewel synth iden.ctx \
		randseed=1 verbose=1 lowmem=1 srctrav=rand crop=random \
		dim=64x64x3 limit=65536 src=/spin/dan/vangogh.xsmall out=$@





%.ctx: %.txt
	tewel make $@ spec=$< clobber=1





normalize: enc0.ctx gen0.ctx

rvg: src1.rvg shampane.rvg nova2.rvg miniceleb.rvg lfw.rvg minidne.rvg cand.rvg

enc0.ctx gen0.ctx: src1.rvg
	tewel normalize enc0.ctx gen0.ctx rvg=src1.rvg verbose=1 clobber=1 nerf=1

enc0-nova2.ctx gen0-nova2.ctx: nova2.rvg
	tewel normalize enc0-nova2.ctx gen0-nova2.ctx rvg=$< verbose=1 clobber=1 nerf=1



src1.rvg: src1.rgb enc1.ctx learn1.done
	tewel synth enc1.ctx dim=128x128 src=$< out=$@ verbose=1

shampane.rvg: dns1.ctx mean.ctx enc1.ctx
	tewel synth dns1.ctx mean.ctx enc1.ctx dim=256x256 src=/spin/dan/shampane.aligned out=$@ verbose=1
nova2.rvg: dns1.ctx mean.ctx enc1.ctx
	tewel synth dns1.ctx mean.ctx enc1.ctx dim=256x256 src=/spin/dan/nova2.aligned out=$@ verbose=1
miniceleb.rvg: dns1.ctx mean.ctx enc1.ctx
	tewel synth dns1.ctx mean.ctx enc1.ctx dim=256x256 src=/spin/dan/miniceleb.aligned out=$@ verbose=1
lfw.rvg: dns1.ctx mean.ctx enc1.ctx
	tewel synth dns1.ctx mean.ctx enc1.ctx dim=256x256 src=/spin/dan/lfw.aligned out=$@ verbose=1
minidne.rvg: dns1.ctx mean.ctx enc1.ctx
	tewel synth dns1.ctx mean.ctx enc1.ctx dim=256x256 src=/spin/dan/minidne.aligned out=$@ verbose=1
cand.rvg: dns1.ctx mean.ctx enc1.ctx
	tewel synth dns1.ctx mean.ctx enc1.ctx dim=256x256 src=/spin/dan/cand.aligned out=$@ verbose=1






show1: enc1.ctx gen1.ctx
	tewel synth enc1.ctx gen1.ctx src=/spin/dan/lq.128 \
		srctrav=rand dim=128x128 repeat=1 reload=1 delay=1 outkind=sdl
show2: enc1.ctx gen1.ctx gen2.ctx
	tewel synth enc1.ctx gen1.ctx gen2.ctx src=/spin/dan/lq.128 \
		srctrav=rand dim=128x128 repeat=1 reload=1 delay=1 outkind=sdl
show3: enc1.ctx gen1.ctx gen2.ctx gen3.ctx
	tewel synth enc1.ctx gen1.ctx gen2.ctx gen3.ctx src=/spin/dan/lq.128 \
		srctrav=rand dim=128x128 repeat=1 reload=1 delay=1 outkind=sdl
show4: enc1.ctx gen1.ctx gen2.ctx gen3.ctx gen4.ctx
	tewel synth enc1.ctx gen1.ctx gen2.ctx gen3.ctx gen4.ctx src=/spin/dan/lq.128 \
		srctrav=rand dim=128x128 repeat=1 reload=1 delay=1 outkind=sdl
shownova: gen1.ctx gen2.ctx gen3.ctx nova2.rvg
	tewel synth gen1.ctx gen2.ctx gen3.ctx src=nova2.rvg \
		evolve=0.95 dim=1x1x1536 repeat=1 reload=1 delay=0.05 outkind=sdl
showpane: gen1.ctx gen2.ctx gen3.ctx shampane.rvg
	tewel synth gen1.ctx gen2.ctx gen3.ctx src=shampane.rvg \
		evolve=0.95 dim=1x1x1536 repeat=1 reload=1 delay=0.05 outkind=sdl


server:
	tewel server gen0-nova2.ctx gen1.ctx gen2.ctx gen3.ctx gen4.ctx verbose=1 dim=1x1x1536 kids=1 reload=5



clean1:
	rm -f gen1.ctx enc1.ctx src1.rgb tgt1.rgb learn1.done
clean2:
	rm -f gen2.ctx dis2.ctx cnd2.ctx $(SRC2) $(TGT2) src2.tmp.u8 tgt2.tmp.u8 learn2.done
clean3:
	rm -f gen3.ctx dis3.ctx cnd3.ctx $(SRC3) $(TGT3) src3.tmp.u8 tgt3.tmp.u8 learn3.done
clean4:
	rm -f gen4.ctx dis4.ctx $(SRC4) tgt4.rgb learn4.done

clean: clean1 clean2 clean3
	rm -f *.rvg *.ctx *.out

"""Rebuild only the seven figures displayed for A0/A1, using shared FigureStyle."""
import redraw_overviews
import redraw_bottleneck
import redraw_a0_a1
import redraw_generator

if __name__=='__main__':
    redraw_overviews.draw('A0')
    redraw_overviews.draw('A1')
    redraw_bottleneck.main()
    redraw_bottleneck.spectral()
    redraw_a0_a1.skip()
    redraw_generator.draw()
    redraw_a0_a1.d2()

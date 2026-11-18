"""CPU-only teaching contract; never imports training code or CUDA."""
import importlib.util
from pathlib import Path
import unittest

class StreamingContract(unittest.TestCase):
    def test_image_tokens_can_be_processed_incrementally(self):
        path=Path(__file__).with_name('stream_demo.py')
        self.assertTrue(path.exists(),'Missing runnable state-streaming demonstration')
        spec=importlib.util.spec_from_file_location('stream_demo',path)
        demo=importlib.util.module_from_spec(spec);spec.loader.exec_module(demo)
        tokens=[0.2,-0.4,0.8,0.1]
        full,final=demo.run_sequence(tokens)
        state=0.0;streamed=[]
        for token in tokens:
            output,state=demo.step(token,state);streamed.append(output)
        self.assertEqual(full,streamed)
        self.assertEqual(final,state)
        first,state=demo.run_sequence(tokens[:2])
        last,state=demo.run_sequence(tokens[2:],state)
        self.assertEqual(full,first+last)
        self.assertEqual(final,state)
        self.assertEqual(demo.run_sequence([]),( [],0.0))
        _,r,j,c=demo.coefficients(tokens[0])
        _,r2,j2,c2=demo.coefficients(tokens[1])
        self.assertAlmostEqual(full[1],c2*(r2*j+j2)+0.5*tokens[1])

if __name__=='__main__':unittest.main(verbosity=2)

from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self,  parser):
        parser = BaseOptions.initialize(self, parser)

        parser.add_argument('--ntest', type=int, default=float("inf"), help='# of the test examples')
        parser.add_argument('--results_dir', type=str, default='result/BDD_OOD/', help='saves results here')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test')
        parser.add_argument('--mask_type', type=int, default=[0],
                            help='mask type, 0: center mask, 1:random regular mask, '
                            '2: random irregular mask from plc (cvpr 2019). 3: external irregular mask. 4: irregular mask from plc (iccv 2019) [0],[1,2],[2,3]')
        parser.add_argument('--save_number', type=int, default=1, help='choice # reasonable results based on the discriminator score')

        self.isTrain = False

        return parser

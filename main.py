from models.Mixture import Mixture
from models.Shift import Shift
from models.Cut import Cut


if __name__ == '__main__':
    mix = Mixture()
    cut = Cut()
    shift = Shift()
    mix.train()
    cut.train()
    shift.train()

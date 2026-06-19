from confounded import create_confounded_scm
from diamond import create_diamond_scm
from collider import create_collider_scm
from confoundedBinary import create_confounded_binary_bn

common_scm = {
    "confounded": create_confounded_scm(),
    "collider": create_collider_scm(),
    "diamond": create_diamond_scm(),
}

common_scm_binary = {
    "confounded": create_confounded_binary_bn(),
}

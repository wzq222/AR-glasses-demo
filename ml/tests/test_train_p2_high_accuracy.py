from ml.scripts.train_p2_high_accuracy import build_argument_parser


def test_training_cli_accepts_all_four_deterministic_tile_views() -> None:
    arguments = build_argument_parser().parse_args(["--train-tile-views", "4"])

    assert arguments.train_tile_views == 4


def test_training_cli_keeps_one_tile_as_the_compatible_default() -> None:
    arguments = build_argument_parser().parse_args([])

    assert arguments.train_tile_views == 1

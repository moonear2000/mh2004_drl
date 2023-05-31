import numpy as np
from hanabi_learning_environment import pyhanabi

def run_random_game(parameters):

    def print_state(state):

        print("")
        print("current player: {}".format(state.cur_player()))
        print(state)
        print("### Information tokens: {}".format(state.information_tokens()))
        print("### Life tokens: {}".format(state.life_tokens()))
        print("### Fireworks: {}".format(state.fireworks()))
        print("### Deck size: {}".format(state.deck_size()))
        print("### Discard pile: {}".format(str(state.discard_pile())))
        print("### Player hands: {}".format(str(state.player_hands())))
        print("")

    def print_observations(observation):
        print("----Observation----")
        print(observation)
        print("### Current player, relative to self: {}".format(
        observation.cur_player_offset()))
        move_string = "### Last moves:"
        for move_tuple in observation.last_moves():
            move_string += " {}".format(move_tuple)
        print(move_string)
        print("### Information tokens: {}".format(observation.information_tokens()))
        print("### Life tokens: {}".format(observation.life_tokens()))
        print("### Legal moves: {}".format(observation.legal_moves()))
        print("--- EndObservation ---")

    def print_encoded_observations(encoder, state, num_players):
        print("--- EncodedObservations ---")
        print("Observation encoding shape: {}".format(encoder.shape()))
        print("Current actual player: {}".format(state.cur_player()))
        for i in range(num_players):
            print("Encoded observation for player {}: {}".format(
            i, encoder.encode(state.observation(i))))
        print("--- EndEncodedObservations ---")

    game = pyhanabi.HanabiGame(parameters)
    print(game.parameter_string(), end="")
    obs_encoder = pyhanabi.ObservationEncoder(
        game, enc_type=pyhanabi.ObservationEncoderType.CANONICAL)

    state = game.new_initial_state()
    while not state.is_terminal():
        if state.cur_player() == pyhanabi.CHANCE_PLAYER_ID:
            state.deal_random_card()
            continue
        print("####STATE#####")
        print_state(state)
        observation = state.observation(state.cur_player())
        print("#####OBSERVATIONS#######")
        print_observations(observation)
        print("####ENCODED OBSERVATIONS#####")
        print_encoded_observations(obs_encoder, state, game.num_players())

        legal_moves = state.legal_moves()
        print("")
        print("Number of legal moves: {}".format(len(legal_moves)))

        move = np.random.choice(legal_moves)
        print("Chose random legal move: {}".format(move))

        state.apply_move(move)
        print("####APPLIED {} #####".format(move))
    
    print("")
    print("Game done. Terminal state:")
    print("")
    print(state)
    print("")
    print("score: {}".format(state.score()))

if __name__ == "__main__":
    assert pyhanabi.cdef_loaded(), "cdef failed to load"
    assert pyhanabi.lib_loaded(), "lib failed to load"
    run_random_game({"players": 2, "random_start_player":True})
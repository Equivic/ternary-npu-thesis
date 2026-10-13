library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity ternary_neuron is 
  port (
    weights   : in std_logic_vector(255 downto 0);   
    inputs    : in std_logic_vector(127 downto 0);
    output    : out std_logic
  );
end ternary_neuron;

architecture structural of ternary_neuron is
  signal contributions : std_logic_vector(255 downto 0);
begin
  gen_pes : for i in 0 to 127 generate
    pe_i : entity work.ternary_pe(dataflow)
      port map(
        weight        =>  weights(2*i+1 downto 2*i),
        input_bit     =>  inputs(i),
        contribution  =>  contributions(2*i+1 downto 2*i)
      );
  end generate;
end structural;

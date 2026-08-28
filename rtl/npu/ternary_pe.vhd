library ieee;
use ieee.std_logic_1164.all;

entity ternary_pe is
    port (
      weight        : in  std_logic_vector( 1 downto 0);
      input_bit     : in  std_logic;
      contribution  : out std_logic_vector( 1 downto 0)
    );
end ternary_pe;

architecture dataflow of ternary_pe is
begin
    contribution(0) <= not weight(1);
    contribution(1) <= (weight(0) xor input_bit) and (not weight(1));
end dataflow;

architecture behavioral of ternary_pe is
begin
    process(weight, input_bit)
    begin
      if weight(1) = '1' then
          contribution <= "00";
      elsif (weight(0) = '1' and input_bit = '1')then
          contribution <= "01";
      elsif (weight(0) = '1' and input_bit = '0')then
          contribution <= "11";
      elsif (weight(0) = '0' and input_bit = '1')then
          contribution <= "11";
      else
          contribution <= "01";
      end if;
    end process;
end behavioral;

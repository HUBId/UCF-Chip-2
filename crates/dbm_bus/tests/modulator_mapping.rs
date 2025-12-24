use biophys_core::ModLevel;
use dbm_bus::modulator_field_from_levels;
use dbm_core::LevelClass;

#[test]
fn reward_block_forces_da_low() {
    let mods = modulator_field_from_levels(
        LevelClass::Med,
        LevelClass::Low,
        LevelClass::High,
        true,
    );
    assert_eq!(mods.da, ModLevel::Low);
    assert_eq!(mods.na, ModLevel::Med);
    assert_eq!(mods.ht, ModLevel::Low);
}

#[test]
fn reward_block_allows_da_progress() {
    let mods = modulator_field_from_levels(
        LevelClass::Low,
        LevelClass::High,
        LevelClass::High,
        false,
    );
    assert_eq!(mods.da, ModLevel::High);
    assert_eq!(mods.ht, ModLevel::High);
}

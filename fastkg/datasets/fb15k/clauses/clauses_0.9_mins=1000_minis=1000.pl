/government/legislative_session/members./government/government_position_held/district_represented(X0, X2) :- /government/legislative_session/members./government/government_position_held/legislative_sessions(X1, X0), /government/political_district/representatives./government/government_position_held/legislative_sessions(X2, X1)
/government/legislative_session/members./government/government_position_held/district_represented(X0, X2) :- /government/legislative_session/members./government/government_position_held/legislative_sessions(X0, X1), /government/political_district/representatives./government/government_position_held/legislative_sessions(X2, X1)
/government/legislative_session/members./government/government_position_held/district_represented(X0, X2) :- /government/legislative_session/members./government/government_position_held/legislative_sessions(X0, X1), /government/legislative_session/members./government/government_position_held/district_represented(X1, X2)
/government/legislative_session/members./government/government_position_held/district_represented(X0, X2) :- /government/legislative_session/members./government/government_position_held/legislative_sessions(X1, X0), /government/legislative_session/members./government/government_position_held/district_represented(X1, X2)
/government/political_district/representatives./government/government_position_held/legislative_sessions(X0, X2) :- /government/political_district/representatives./government/government_position_held/legislative_sessions(X0, X1), /government/legislative_session/members./government/government_position_held/legislative_sessions(X2, X1)
/government/political_district/representatives./government/government_position_held/legislative_sessions(X0, X2) :- /government/legislative_session/members./government/government_position_held/district_represented(X1, X0), /government/legislative_session/members./government/government_position_held/legislative_sessions(X2, X1)
/government/political_district/representatives./government/government_position_held/legislative_sessions(X0, X2) :- /government/political_district/representatives./government/government_position_held/legislative_sessions(X0, X1), /government/legislative_session/members./government/government_position_held/legislative_sessions(X1, X2)
/government/political_district/representatives./government/government_position_held/legislative_sessions(X0, X2) :- /government/legislative_session/members./government/government_position_held/district_represented(X1, X0), /government/legislative_session/members./government/government_position_held/legislative_sessions(X1, X2)
/olympics/olympic_sport/athletes./olympics/olympic_athlete_affiliation/country(X0, X2) :- /olympics/olympic_games/athletes./olympics/olympic_athlete_affiliation/sport(X1, X0), /olympics/olympic_participating_country/athletes./olympics/olympic_athlete_affiliation/olympics(X2, X1)
/olympics/olympic_sport/athletes./olympics/olympic_athlete_affiliation/country(X0, X2) :- /olympics/olympic_games/athletes./olympics/olympic_athlete_affiliation/sport(X1, X0), /olympics/olympic_games/athletes./olympics/olympic_athlete_affiliation/country(X1, X2)
/music/performance_role/regular_performances./music/group_membership/role(X0, X2) :- /music/performance_role/regular_performances./music/group_membership/role(X1, X0), /music/performance_role/track_performances./music/track_contribution/role(X2, X1)
/music/performance_role/regular_performances./music/group_membership/role(X0, X2) :- /music/performance_role/regular_performances./music/group_membership/role(X0, X1), /music/performance_role/track_performances./music/track_contribution/role(X2, X1)
/music/performance_role/regular_performances./music/group_membership/role(X0, X2) :- /music/performance_role/regular_performances./music/group_membership/role(X1, X0), /music/performance_role/regular_performances./music/group_membership/role(X2, X1)
/music/performance_role/regular_performances./music/group_membership/role(X0, X2) :- /music/performance_role/track_performances./music/track_contribution/role(X1, X0), /music/performance_role/track_performances./music/track_contribution/role(X2, X1)
/music/performance_role/regular_performances./music/group_membership/role(X0, X2) :- /music/performance_role/track_performances./music/track_contribution/role(X0, X1), /music/performance_role/track_performances./music/track_contribution/role(X2, X1)
/music/performance_role/regular_performances./music/group_membership/role(X0, X2) :- /music/performance_role/track_performances./music/track_contribution/role(X1, X0), /music/performance_role/regular_performances./music/group_membership/role(X2, X1)
/music/performance_role/regular_performances./music/group_membership/role(X0, X2) :- /music/performance_role/regular_performances./music/group_membership/role(X0, X1), /music/performance_role/regular_performances./music/group_membership/role(X2, X1)
/music/performance_role/regular_performances./music/group_membership/role(X0, X2) :- /music/performance_role/regular_performances./music/group_membership/role(X0, X1), /music/performance_role/track_performances./music/track_contribution/role(X1, X2)
/music/performance_role/regular_performances./music/group_membership/role(X0, X2) :- /music/performance_role/regular_performances./music/group_membership/role(X0, X1), /music/performance_role/regular_performances./music/group_membership/role(X1, X2)
/music/performance_role/regular_performances./music/group_membership/role(X0, X2) :- /music/performance_role/track_performances./music/track_contribution/role(X0, X1), /music/performance_role/regular_performances./music/group_membership/role(X2, X1)
/music/performance_role/regular_performances./music/group_membership/role(X0, X2) :- /music/performance_role/track_performances./music/track_contribution/role(X0, X1), /music/performance_role/regular_performances./music/group_membership/role(X1, X2)
/music/performance_role/regular_performances./music/group_membership/role(X0, X2) :- /music/performance_role/regular_performances./music/group_membership/role(X1, X0), /music/performance_role/regular_performances./music/group_membership/role(X1, X2)
/music/performance_role/regular_performances./music/group_membership/role(X0, X2) :- /music/performance_role/track_performances./music/track_contribution/role(X1, X0), /music/performance_role/regular_performances./music/group_membership/role(X1, X2)
/music/performance_role/regular_performances./music/group_membership/role(X0, X2) :- /music/performance_role/regular_performances./music/group_membership/role(X1, X0), /music/performance_role/track_performances./music/track_contribution/role(X1, X2)
/music/performance_role/regular_performances./music/group_membership/role(X0, X2) :- /music/performance_role/track_performances./music/track_contribution/role(X0, X1), /music/performance_role/track_performances./music/track_contribution/role(X1, X2)
/music/performance_role/regular_performances./music/group_membership/role(X0, X2) :- /music/performance_role/track_performances./music/track_contribution/role(X1, X0), /music/performance_role/track_performances./music/track_contribution/role(X1, X2)
/olympics/olympic_participating_country/medals_won./olympics/olympic_medal_honor/olympics(X0, X2) :- /olympics/olympic_medal/medal_winners./olympics/olympic_medal_honor/country(X1, X0), /olympics/olympic_games/medals_awarded./olympics/olympic_medal_honor/medal(X2, X1)
/olympics/olympic_participating_country/medals_won./olympics/olympic_medal_honor/olympics(X0, X2) :- /olympics/olympic_participating_country/medals_won./olympics/olympic_medal_honor/medal(X0, X1), /olympics/olympic_games/medals_awarded./olympics/olympic_medal_honor/medal(X2, X1)
/olympics/olympic_participating_country/medals_won./olympics/olympic_medal_honor/olympics(X0, X2) :- /olympics/olympic_sport/athletes./olympics/olympic_athlete_affiliation/country(X1, X0), /olympics/olympic_games/sports(X2, X1)
/olympics/olympic_participating_country/medals_won./olympics/olympic_medal_honor/olympics(X0, X2) :- /olympics/olympic_participating_country/athletes./olympics/olympic_athlete_affiliation/sport(X0, X1), /olympics/olympic_games/sports(X2, X1)
/olympics/olympic_participating_country/medals_won./olympics/olympic_medal_honor/olympics(X0, X2) :- /olympics/olympic_participating_country/medals_won./olympics/olympic_medal_honor/medal(X0, X1), /olympics/olympic_medal/medal_winners./olympics/olympic_medal_honor/olympics(X1, X2)
/olympics/olympic_participating_country/medals_won./olympics/olympic_medal_honor/olympics(X0, X2) :- /olympics/olympic_sport/athletes./olympics/olympic_athlete_affiliation/country(X1, X0), /olympics/olympic_games/athletes./olympics/olympic_athlete_affiliation/sport(X2, X1)
/olympics/olympic_participating_country/medals_won./olympics/olympic_medal_honor/olympics(X0, X2) :- /olympics/olympic_participating_country/athletes./olympics/olympic_athlete_affiliation/sport(X0, X1), /olympics/olympic_games/athletes./olympics/olympic_athlete_affiliation/sport(X2, X1)
/olympics/olympic_participating_country/medals_won./olympics/olympic_medal_honor/olympics(X0, X2) :- /olympics/olympic_participating_country/athletes./olympics/olympic_athlete_affiliation/sport(X0, X1), /olympics/olympic_sport/athletes./olympics/olympic_athlete_affiliation/olympics(X1, X2)
/olympics/olympic_participating_country/medals_won./olympics/olympic_medal_honor/olympics(X0, X2) :- /olympics/olympic_participating_country/athletes./olympics/olympic_athlete_affiliation/sport(X0, X1), /olympics/olympic_sport/olympic_games_contested(X1, X2)
/olympics/olympic_participating_country/medals_won./olympics/olympic_medal_honor/olympics(X0, X2) :- /olympics/olympic_medal/medal_winners./olympics/olympic_medal_honor/country(X1, X0), /olympics/olympic_medal/medal_winners./olympics/olympic_medal_honor/olympics(X1, X2)
/olympics/olympic_participating_country/medals_won./olympics/olympic_medal_honor/olympics(X0, X2) :- /olympics/olympic_sport/athletes./olympics/olympic_athlete_affiliation/country(X1, X0), /olympics/olympic_sport/athletes./olympics/olympic_athlete_affiliation/olympics(X1, X2)
/olympics/olympic_participating_country/medals_won./olympics/olympic_medal_honor/olympics(X0, X2) :- /olympics/olympic_sport/athletes./olympics/olympic_athlete_affiliation/country(X1, X0), /olympics/olympic_sport/olympic_games_contested(X1, X2)
/soccer/football_team/current_roster./sports/sports_team_roster/position(X0, X2) :- /soccer/football_team/current_roster./sports/sports_team_roster/position(X0, X1), /sports/sports_position/players./sports/sports_team_roster/position(X1, X2)
/soccer/football_team/current_roster./sports/sports_team_roster/position(X0, X2) :- /soccer/football_team/current_roster./sports/sports_team_roster/position(X0, X1), /sports/sports_position/players./sports/sports_team_roster/position(X2, X1)
/soccer/football_team/current_roster./sports/sports_team_roster/position(X0, X2) :- /sports/sports_position/players./soccer/football_roster_position/team(X1, X0), /sports/sports_position/players./sports/sports_team_roster/position(X2, X1)
/soccer/football_team/current_roster./sports/sports_team_roster/position(X0, X2) :- /sports/sports_position/players./sports/sports_team_roster/team(X1, X0), /sports/sports_position/players./sports/sports_team_roster/position(X2, X1)
/soccer/football_team/current_roster./sports/sports_team_roster/position(X0, X2) :- /sports/sports_team/roster./soccer/football_roster_position/position(X0, X1), /sports/sports_position/players./sports/sports_team_roster/position(X2, X1)
/soccer/football_team/current_roster./sports/sports_team_roster/position(X0, X2) :- /soccer/football_team/current_roster./soccer/football_roster_position/position(X0, X1), /sports/sports_position/players./sports/sports_team_roster/position(X2, X1)
/soccer/football_team/current_roster./sports/sports_team_roster/position(X0, X2) :- /sports/sports_team/roster./sports/sports_team_roster/position(X0, X1), /sports/sports_position/players./sports/sports_team_roster/position(X2, X1)
/soccer/football_team/current_roster./sports/sports_team_roster/position(X0, X2) :- /sports/sports_team/roster./soccer/football_roster_position/position(X0, X1), /sports/sports_position/players./sports/sports_team_roster/position(X1, X2)
/soccer/football_team/current_roster./sports/sports_team_roster/position(X0, X2) :- /sports/sports_team/roster./sports/sports_team_roster/position(X0, X1), /sports/sports_position/players./sports/sports_team_roster/position(X1, X2)
/soccer/football_team/current_roster./sports/sports_team_roster/position(X0, X2) :- /soccer/football_team/current_roster./soccer/football_roster_position/position(X0, X1), /sports/sports_position/players./sports/sports_team_roster/position(X1, X2)
/soccer/football_team/current_roster./sports/sports_team_roster/position(X0, X2) :- /sports/sports_position/players./soccer/football_roster_position/team(X1, X0), /sports/sports_position/players./sports/sports_team_roster/position(X1, X2)
/soccer/football_team/current_roster./sports/sports_team_roster/position(X0, X2) :- /sports/sports_position/players./sports/sports_team_roster/team(X1, X0), /sports/sports_position/players./sports/sports_team_roster/position(X1, X2)
/sports/sports_position/players./soccer/football_roster_position/team(X0, X2) :- /sports/sports_position/players./sports/sports_team_roster/position(X1, X0), /sports/sports_team/roster./soccer/football_roster_position/position(X2, X1)
/sports/sports_position/players./soccer/football_roster_position/team(X0, X2) :- /sports/sports_position/players./sports/sports_team_roster/position(X0, X1), /sports/sports_team/roster./soccer/football_roster_position/position(X2, X1)
/sports/sports_position/players./soccer/football_roster_position/team(X0, X2) :- /sports/sports_position/players./sports/sports_team_roster/position(X1, X0), /sports/sports_team/roster./sports/sports_team_roster/position(X2, X1)
/sports/sports_position/players./soccer/football_roster_position/team(X0, X2) :- /sports/sports_position/players./sports/sports_team_roster/position(X0, X1), /sports/sports_team/roster./sports/sports_team_roster/position(X2, X1)
/sports/sports_position/players./soccer/football_roster_position/team(X0, X2) :- /sports/sports_position/players./sports/sports_team_roster/position(X1, X0), /soccer/football_team/current_roster./sports/sports_team_roster/position(X2, X1)
/sports/sports_position/players./soccer/football_roster_position/team(X0, X2) :- /sports/sports_position/players./sports/sports_team_roster/position(X0, X1), /soccer/football_team/current_roster./sports/sports_team_roster/position(X2, X1)
/sports/sports_position/players./soccer/football_roster_position/team(X0, X2) :- /sports/sports_position/players./sports/sports_team_roster/position(X1, X0), /soccer/football_team/current_roster./soccer/football_roster_position/position(X2, X1)
/sports/sports_position/players./soccer/football_roster_position/team(X0, X2) :- /sports/sports_position/players./sports/sports_team_roster/position(X0, X1), /soccer/football_team/current_roster./soccer/football_roster_position/position(X2, X1)
/sports/sports_position/players./soccer/football_roster_position/team(X0, X2) :- /sports/sports_position/players./sports/sports_team_roster/position(X0, X1), /sports/sports_position/players./soccer/football_roster_position/team(X1, X2)
/sports/sports_position/players./soccer/football_roster_position/team(X0, X2) :- /sports/sports_position/players./sports/sports_team_roster/position(X0, X1), /sports/sports_position/players./sports/sports_team_roster/team(X1, X2)
/sports/sports_position/players./soccer/football_roster_position/team(X0, X2) :- /sports/sports_position/players./sports/sports_team_roster/position(X1, X0), /sports/sports_position/players./soccer/football_roster_position/team(X1, X2)
/sports/sports_position/players./soccer/football_roster_position/team(X0, X2) :- /sports/sports_position/players./sports/sports_team_roster/position(X1, X0), /sports/sports_position/players./sports/sports_team_roster/team(X1, X2)
/olympics/olympic_participating_country/athletes./olympics/olympic_athlete_affiliation/sport(X0, X2) :- /olympics/olympic_participating_country/athletes./olympics/olympic_athlete_affiliation/olympics(X0, X1), /olympics/olympic_games/athletes./olympics/olympic_athlete_affiliation/sport(X1, X2)
/music/performance_role/track_performances./music/track_contribution/role(X0, X2) :- /music/performance_role/regular_performances./music/group_membership/role(X1, X0), /music/performance_role/regular_performances./music/group_membership/role(X2, X1)
/music/performance_role/track_performances./music/track_contribution/role(X0, X2) :- /music/performance_role/track_performances./music/track_contribution/role(X1, X0), /music/performance_role/regular_performances./music/group_membership/role(X2, X1)
/music/performance_role/track_performances./music/track_contribution/role(X0, X2) :- /music/performance_role/track_performances./music/track_contribution/role(X0, X1), /music/performance_role/regular_performances./music/group_membership/role(X2, X1)
/music/performance_role/track_performances./music/track_contribution/role(X0, X2) :- /music/performance_role/regular_performances./music/group_membership/role(X0, X1), /music/performance_role/track_performances./music/track_contribution/role(X2, X1)
/music/performance_role/track_performances./music/track_contribution/role(X0, X2) :- /music/performance_role/regular_performances./music/group_membership/role(X0, X1), /music/performance_role/track_performances./music/track_contribution/role(X1, X2)
/music/performance_role/track_performances./music/track_contribution/role(X0, X2) :- /music/performance_role/track_performances./music/track_contribution/role(X1, X0), /music/performance_role/track_performances./music/track_contribution/role(X2, X1)
/music/performance_role/track_performances./music/track_contribution/role(X0, X2) :- /music/performance_role/regular_performances./music/group_membership/role(X0, X1), /music/performance_role/regular_performances./music/group_membership/role(X1, X2)
/music/performance_role/track_performances./music/track_contribution/role(X0, X2) :- /music/performance_role/track_performances./music/track_contribution/role(X0, X1), /music/performance_role/track_performances./music/track_contribution/role(X2, X1)
/music/performance_role/track_performances./music/track_contribution/role(X0, X2) :- /music/performance_role/track_performances./music/track_contribution/role(X0, X1), /music/performance_role/track_performances./music/track_contribution/role(X1, X2)
/music/performance_role/track_performances./music/track_contribution/role(X0, X2) :- /music/performance_role/track_performances./music/track_contribution/role(X0, X1), /music/performance_role/regular_performances./music/group_membership/role(X1, X2)
/music/performance_role/track_performances./music/track_contribution/role(X0, X2) :- /music/performance_role/track_performances./music/track_contribution/role(X1, X0), /music/performance_role/regular_performances./music/group_membership/role(X1, X2)
/music/performance_role/track_performances./music/track_contribution/role(X0, X2) :- /music/performance_role/regular_performances./music/group_membership/role(X1, X0), /music/performance_role/regular_performances./music/group_membership/role(X1, X2)
/music/performance_role/track_performances./music/track_contribution/role(X0, X2) :- /music/performance_role/regular_performances./music/group_membership/role(X1, X0), /music/performance_role/track_performances./music/track_contribution/role(X1, X2)
/music/performance_role/track_performances./music/track_contribution/role(X0, X2) :- /music/performance_role/track_performances./music/track_contribution/role(X1, X0), /music/performance_role/track_performances./music/track_contribution/role(X1, X2)
/soccer/football_team/current_roster./soccer/football_roster_position/position(X0, X2) :- /sports/sports_position/players./soccer/football_roster_position/team(X1, X0), /sports/sports_position/players./sports/sports_team_roster/position(X2, X1)
/soccer/football_team/current_roster./soccer/football_roster_position/position(X0, X2) :- /sports/sports_position/players./sports/sports_team_roster/team(X1, X0), /sports/sports_position/players./sports/sports_team_roster/position(X2, X1)
/soccer/football_team/current_roster./soccer/football_roster_position/position(X0, X2) :- /sports/sports_team/roster./soccer/football_roster_position/position(X0, X1), /sports/sports_position/players./sports/sports_team_roster/position(X2, X1)
/soccer/football_team/current_roster./soccer/football_roster_position/position(X0, X2) :- /soccer/football_team/current_roster./soccer/football_roster_position/position(X0, X1), /sports/sports_position/players./sports/sports_team_roster/position(X2, X1)
/soccer/football_team/current_roster./soccer/football_roster_position/position(X0, X2) :- /soccer/football_team/current_roster./sports/sports_team_roster/position(X0, X1), /sports/sports_position/players./sports/sports_team_roster/position(X2, X1)
/soccer/football_team/current_roster./soccer/football_roster_position/position(X0, X2) :- /sports/sports_team/roster./sports/sports_team_roster/position(X0, X1), /sports/sports_position/players./sports/sports_team_roster/position(X2, X1)
/soccer/football_team/current_roster./soccer/football_roster_position/position(X0, X2) :- /soccer/football_team/current_roster./soccer/football_roster_position/position(X0, X1), /sports/sports_position/players./sports/sports_team_roster/position(X1, X2)
/soccer/football_team/current_roster./soccer/football_roster_position/position(X0, X2) :- /sports/sports_team/roster./soccer/football_roster_position/position(X0, X1), /sports/sports_position/players./sports/sports_team_roster/position(X1, X2)
/soccer/football_team/current_roster./soccer/football_roster_position/position(X0, X2) :- /sports/sports_team/roster./sports/sports_team_roster/position(X0, X1), /sports/sports_position/players./sports/sports_team_roster/position(X1, X2)
/soccer/football_team/current_roster./soccer/football_roster_position/position(X0, X2) :- /soccer/football_team/current_roster./sports/sports_team_roster/position(X0, X1), /sports/sports_position/players./sports/sports_team_roster/position(X1, X2)
/soccer/football_team/current_roster./soccer/football_roster_position/position(X0, X2) :- /sports/sports_position/players./soccer/football_roster_position/team(X1, X0), /sports/sports_position/players./sports/sports_team_roster/position(X1, X2)
/soccer/football_team/current_roster./soccer/football_roster_position/position(X0, X2) :- /sports/sports_position/players./sports/sports_team_roster/team(X1, X0), /sports/sports_position/players./sports/sports_team_roster/position(X1, X2)
/sports/sports_team/roster./soccer/football_roster_position/position(X0, X2) :- /sports/sports_position/players./soccer/football_roster_position/team(X1, X0), /sports/sports_position/players./sports/sports_team_roster/position(X2, X1)
/sports/sports_team/roster./soccer/football_roster_position/position(X0, X2) :- /sports/sports_position/players./sports/sports_team_roster/team(X1, X0), /sports/sports_position/players./sports/sports_team_roster/position(X2, X1)
/sports/sports_team/roster./soccer/football_roster_position/position(X0, X2) :- /sports/sports_team/roster./soccer/football_roster_position/position(X0, X1), /sports/sports_position/players./sports/sports_team_roster/position(X2, X1)
/sports/sports_team/roster./soccer/football_roster_position/position(X0, X2) :- /soccer/football_team/current_roster./sports/sports_team_roster/position(X0, X1), /sports/sports_position/players./sports/sports_team_roster/position(X2, X1)
/sports/sports_team/roster./soccer/football_roster_position/position(X0, X2) :- /soccer/football_team/current_roster./soccer/football_roster_position/position(X0, X1), /sports/sports_position/players./sports/sports_team_roster/position(X2, X1)
/sports/sports_team/roster./soccer/football_roster_position/position(X0, X2) :- /sports/sports_team/roster./sports/sports_team_roster/position(X0, X1), /sports/sports_position/players./sports/sports_team_roster/position(X2, X1)
/sports/sports_team/roster./soccer/football_roster_position/position(X0, X2) :- /soccer/football_team/current_roster./sports/sports_team_roster/position(X0, X1), /sports/sports_position/players./sports/sports_team_roster/position(X1, X2)
/sports/sports_team/roster./soccer/football_roster_position/position(X0, X2) :- /sports/sports_team/roster./soccer/football_roster_position/position(X0, X1), /sports/sports_position/players./sports/sports_team_roster/position(X1, X2)
/sports/sports_team/roster./soccer/football_roster_position/position(X0, X2) :- /sports/sports_team/roster./sports/sports_team_roster/position(X0, X1), /sports/sports_position/players./sports/sports_team_roster/position(X1, X2)
/sports/sports_team/roster./soccer/football_roster_position/position(X0, X2) :- /soccer/football_team/current_roster./soccer/football_roster_position/position(X0, X1), /sports/sports_position/players./sports/sports_team_roster/position(X1, X2)
/sports/sports_team/roster./soccer/football_roster_position/position(X0, X2) :- /sports/sports_position/players./soccer/football_roster_position/team(X1, X0), /sports/sports_position/players./sports/sports_team_roster/position(X1, X2)
/sports/sports_team/roster./soccer/football_roster_position/position(X0, X2) :- /sports/sports_position/players./sports/sports_team_roster/team(X1, X0), /sports/sports_position/players./sports/sports_team_roster/position(X1, X2)
/olympics/olympic_games/medals_awarded./olympics/olympic_medal_honor/country(X0, X2) :- /olympics/olympic_medal/medal_winners./olympics/olympic_medal_honor/olympics(X1, X0), /olympics/olympic_participating_country/medals_won./olympics/olympic_medal_honor/medal(X2, X1)
/olympics/olympic_games/medals_awarded./olympics/olympic_medal_honor/country(X0, X2) :- /olympics/olympic_games/medals_awarded./olympics/olympic_medal_honor/medal(X0, X1), /olympics/olympic_participating_country/medals_won./olympics/olympic_medal_honor/medal(X2, X1)
/olympics/olympic_games/medals_awarded./olympics/olympic_medal_honor/country(X0, X2) :- /olympics/olympic_sport/athletes./olympics/olympic_athlete_affiliation/olympics(X1, X0), /olympics/olympic_participating_country/athletes./olympics/olympic_athlete_affiliation/sport(X2, X1)
/olympics/olympic_games/medals_awarded./olympics/olympic_medal_honor/country(X0, X2) :- /olympics/olympic_sport/olympic_games_contested(X1, X0), /olympics/olympic_participating_country/athletes./olympics/olympic_athlete_affiliation/sport(X2, X1)
/olympics/olympic_games/medals_awarded./olympics/olympic_medal_honor/country(X0, X2) :- /olympics/olympic_games/athletes./olympics/olympic_athlete_affiliation/sport(X0, X1), /olympics/olympic_participating_country/athletes./olympics/olympic_athlete_affiliation/sport(X2, X1)
/olympics/olympic_games/medals_awarded./olympics/olympic_medal_honor/country(X0, X2) :- /olympics/olympic_games/sports(X0, X1), /olympics/olympic_participating_country/athletes./olympics/olympic_athlete_affiliation/sport(X2, X1)
/olympics/olympic_games/medals_awarded./olympics/olympic_medal_honor/country(X0, X2) :- /olympics/olympic_games/medals_awarded./olympics/olympic_medal_honor/medal(X0, X1), /olympics/olympic_medal/medal_winners./olympics/olympic_medal_honor/country(X1, X2)
/olympics/olympic_games/medals_awarded./olympics/olympic_medal_honor/country(X0, X2) :- /olympics/olympic_games/athletes./olympics/olympic_athlete_affiliation/sport(X0, X1), /olympics/olympic_sport/athletes./olympics/olympic_athlete_affiliation/country(X1, X2)
/olympics/olympic_games/medals_awarded./olympics/olympic_medal_honor/country(X0, X2) :- /olympics/olympic_games/sports(X0, X1), /olympics/olympic_sport/athletes./olympics/olympic_athlete_affiliation/country(X1, X2)
/olympics/olympic_games/medals_awarded./olympics/olympic_medal_honor/country(X0, X2) :- /olympics/olympic_medal/medal_winners./olympics/olympic_medal_honor/olympics(X1, X0), /olympics/olympic_medal/medal_winners./olympics/olympic_medal_honor/country(X1, X2)
/olympics/olympic_games/medals_awarded./olympics/olympic_medal_honor/country(X0, X2) :- /olympics/olympic_sport/athletes./olympics/olympic_athlete_affiliation/olympics(X1, X0), /olympics/olympic_sport/athletes./olympics/olympic_athlete_affiliation/country(X1, X2)
/olympics/olympic_games/medals_awarded./olympics/olympic_medal_honor/country(X0, X2) :- /olympics/olympic_sport/olympic_games_contested(X1, X0), /olympics/olympic_sport/athletes./olympics/olympic_athlete_affiliation/country(X1, X2)

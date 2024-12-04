package com.library.booksystem.repository;

import com.library.booksystem.model.InvalidedToken;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface InvalidedTokenRepository extends JpaRepository<InvalidedToken, String> {
}

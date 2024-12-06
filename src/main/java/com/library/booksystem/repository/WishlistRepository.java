package com.library.booksystem.repository;

import com.library.booksystem.model.Book;
import com.library.booksystem.model.Wishlist;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface WishlistRepository extends JpaRepository<Wishlist, Integer> {
    @Query("SELECT w.book FROM Wishlist w WHERE user.userId = :userId")
    List<Book> findWishlistByUserId(@Param("userId") String userId);
}
